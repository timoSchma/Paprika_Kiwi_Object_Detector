import keras
from keras import backend as K
from keras.layers import Conv2D, Input, UpSampling2D, MaxPooling2D, Concatenate, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model

import numpy as np
import sklearn.model_selection
from enum import Enum
import copy
from functools import reduce

from src.model_utils import yolo_eval, yolo_loss
from src.annotated_image import AnnotatedImage, Image
from src.visualization import RGBColors
from src.utils import Dimension
import src.generator


class Mode(Enum):
    """The current mode of the model."""
    STANDARD = 0
    TRAINING = 1


class YoloConfig:
    """Helper class to store the YOLO configuration (i.e. input size, classes, anchors,...)"""
    class_names = ["Paprika", "Kiwi"]
    class_colors = {
        "Paprika": RGBColors.RED,
        "Kiwi": RGBColors.GREEN
    }

    grid = Dimension(13, 13)
    input_size = Dimension(416, 416)
    anchors = np.array([[10, 14],  [23, 27],  [37, 58],  [81, 82],  [135, 169],  [344, 319]], dtype=float)
    scales = {0: 32, 1: 16}
    channels = 3

    optimizer = keras.optimizers.Adam
    learning_rate = 1e-3
    ignore_thresh = 0.7

    score = 0.2
    iou = 0.2

    def num_classes(self):
        """
        Returns the number of classes.

        Returns
        -------
        int
            The number of classes.
        """
        return len(self.class_names)

    def num_output_layers(self):
        """
        Returns the number of output/yolo layers.

        Returns
        -------
        int
            The number of output/yolo layers.
        """
        return len(self.scales)

    def num_anchors(self):
        """
        Returns the number of anchors per output layer.

        Returns
        -------
        int
            The number of anchors per output layer.
        """
        return int(len(self.anchors) / self.num_output_layers())


def train_test_split(annotations, validation_split, random_state=None):
    """
    Perform a train test split on the given dict of annotation.

    Parameters
    ----------
    annotations : dict
        A dict of annotations describing the dataset.
    validation_split : double
        The relative size of the validation set (e.g. 0.2 for 80% train, 20% validation data).
    random_state : int
        Optional random state for deterministic repeatability.

    Returns
    -------
    training_data : dict
        The training data (as dict of annotations)
    validation_data : dict
        The validation data (as dict of annotations)
    """
    indices = list(range(len(annotations)))
    training_indices, validation_indices = sklearn.model_selection.train_test_split(
        indices, train_size=1 - validation_split, test_size=validation_split, random_state=random_state)

    training_data = {key: annotations[key] for i, key in enumerate(annotations)
                     if i in training_indices}
    validation_data = {key: annotations[key] for i, key in enumerate(annotations)
                       if i in validation_indices}

    return training_data, validation_data


class TinyYoloV3:
    """A Tiny-YOLO v3 model with functionality for training (incl. freezing & bottleneck training) and inference."""

    output_layers = 4
    darknet_body_layers = 42

    bottleneck_out_indices = [40, 41]
    last_layer_ids = [[42, 44], [43, 45]]

    def __init__(self, config=YoloConfig(), path=None, pre_trained_weights=False):
        """
        Create a new Tiny-YOLO v3 model with the given configuration.

        Parameters
        ----------
        config : YoloConfig
            The configuration to use, if none is given, the default configuration (see YoloConfig) is used.
        path : path-like
            Optional path to load the weights from.
        """
        self.config = config

        self.mode = Mode.STANDARD

        if pre_trained_weights:
            self.model = self.make_model(self.config, num_classes=80)
        else:
            self.model = self.make_model(self.config)

        if path is not None:
            self.load_weights(path)

        self.original_input, self.original_output = None, None
        self.update_original_in_and_output_layers()

        self.input_image_shape = K.placeholder(shape=(2, ))

        self.session = None
        self.boxes, self.scores, self.classes = None, None, None

        self.compiled = False

    def create_bottleneck_and_last_layer_model(self):
        """
        Creates a new bottleneck and last-layer model used for bottleneck training.

        Returns
        -------
        bottleneck_model : keras.models.Model
            The bottleneck model.
        last_layer_model : keras.models.Model
            The last-layer model.
        """
        # Create bottleneck model
        y_true_layers = self.construct_y_true_layers()
        bottleneck_outputs = [self.model.layers[i].output for i in self.bottleneck_out_indices]
        bottleneck_model = Model([self.original_input, *y_true_layers], bottleneck_outputs)

        # Create last layer model
        # Receives the bottleneck features as inputs
        # Consists of the last/yolo layers (last 4 layers in this model)
        last_layer_inputs = [Input(shape=feature.shape[1:].as_list()) for feature in bottleneck_outputs]
        last_layers = [[self.model.layers[i] for i in ids] for ids in self.last_layer_ids]

        last_layer_outputs = [reduce(lambda inputs, layer: layer(inputs), layers, inputs)
                              for layers, inputs in zip(last_layers, last_layer_inputs)]

        last_layer_model = Model(inputs=last_layer_inputs, outputs=last_layer_outputs)
        last_layer_model = self.wrap_model_with_y_true_and_yolo_loss(last_layer_model)

        return bottleneck_model, last_layer_model

    def __prepare_model_for_inference(self):
        """
        Prepares this model for inference for setting the mode to standard (removes potential wrapper-layers from
        training), setting the session and boxes, scores and classes ''predictors''.
        """
        self.set_mode(Mode.STANDARD)
        self.session = K.get_session()
        _eval = yolo_eval(self.original_output, self.config.anchors,
                          self.config.num_classes(), self.input_image_shape,
                          score_threshold=self.config.score,
                          iou_threshold=self.config.iou)
        self.boxes, self.scores, self.classes = _eval

    def __prepare_image_for_inference(self, image):
        """
        Prepares the given image for inference by resizing it to the expected input size (defined in the model
        configuration) and rescaling the RGB values to [0.0, 1.0].

        Parameters
        ----------
        image : Union[annotated_image.Image, np.array]
            The input image to prepare.
        Returns
        -------
        image_data : np.array
            The prepared image.
        original_shape : tuple
            The original shape of the input image.
        """
        # Handle images both as annotated_image.Image and as np.array
        if isinstance(image, Image):
            original_shape = image.image_data.shape
            image = copy.deepcopy(image)
        elif isinstance(image, np.ndarray):
            original_shape = image.shape
            image = Image(image.copy())
        else:
            raise ValueError(f"Invalid image type {type(image)}")

        # Scale to input size expected by the model
        image.resize_keep_aspect_ratio(*self.config.input_size)

        # Convert to array and normalize from [0, 255] to [0.0, 1.0]
        image_data = np.array(image.image_data, dtype='float32') / 255
        # Expand to array containing one image
        image_data = np.expand_dims(image_data, 0)
        return image_data, original_shape

    def predict(self, image):
        """
        Perform object detection on the given image and returns the raw yolo-prediction.
        Use detect instead to interpret the prediction and receive an annotated image.

        Parameters
        ----------
        image : np.array or Image
            The input image to predict on.

        Returns
        -------
        prediction : List of 3-tuples (box, score, class_id) : (np.array, double, int)
            The prediction made by the yolo model for the given input image.
        """
        # prepare the model for inference
        if self.session is None:
            self.__prepare_model_for_inference()

        # prepare the image for inference (scale, convert to float array,...)
        image_data, original_shape = self.__prepare_image_for_inference(image)

        # make a prediction and return it
        feed_dict = {self.original_input: image_data,
                     self.input_image_shape: original_shape[0:2],
                     K.learning_phase(): 0}
        prediction = self.session.run([self.boxes, self.scores, self.classes], feed_dict=feed_dict)
        return prediction

    def detect(self, image, show=True):
        """
        Perform object detection on the given image, annotate the image with the predicted bounding boxes and labels.

        Parameters
        ----------
        image : np.array or Image
            The input image to predict on.
        show : boolean
            Whether to display or just return the prediction.

        Returns
        -------
        annotated_image : AnnotatedImage
            The prediction as annotated image.
        """
        prediction = self.predict(image)

        # convert prediction to annotated image
        input_image = image.image_data if isinstance(image, Image) else image
        annotated_image = AnnotatedImage.from_yolo_prediction(prediction, input_image, self.config.class_names)

        # show and return annotated image
        if show:
            annotated_image.show(self.config.class_colors)
        return annotated_image

    def set_mode(self, mode):
        """
        Set the current model mode to the given mode. Does nothing if the new mode is invalid or the same as the
        current mode.
        When changing to training model, the model is wrapped with the Loss layer.
        When changing back to standard mode, this wrapping is undone using the original in and output layers.

        Parameters
        ----------
        mode : Mode
            The new mode to set.
        """
        if self.mode is mode or mode not in [Mode.TRAINING, Mode.STANDARD]:
            return
        if mode is Mode.TRAINING:
            self.update_original_in_and_output_layers()
            self.model = self.wrap_model_with_y_true_and_yolo_loss(self.model)
        else:  # Mode.STANDARD
            self.model = keras.models.Model(self.original_input, self.original_output)
        self.mode = mode

    def update_original_in_and_output_layers(self):
        """Update the reference to the original in and output layer with the model's current in and output layers."""
        self.original_input = self.model.input
        self.original_output = self.model.output

    def construct_y_true_layers(self):
        """
        Constructs a new y_true input layer for the ground truth annotations.

        Returns
        -------
        list
            A list of new input layers for each of the output layers/grid shapes.
        """
        # Define input layer for ground truth annotation
        y_true_shapes = self.get_y_shapes(self.config)
        return [keras.layers.Input(shape=shape) for shape in y_true_shapes]

    def wrap_model_with_y_true_and_yolo_loss(self, model):
        """
        Wrap the given model with an additional y_true input layer and an additional YOLO loss output layer.

        Parameters
        ----------
        model : keras.models.Model
            The model to wrap with the additional in and output layers.

        Returns
        -------
        model : keras.models.Model
            The wrapped model.
        """
        # get in- and output layers of the given model and convert them to list format (to avoid special cases)
        original_inputs = model.input if type(model.input) is list else [model.input]
        original_outputs = model.output if type(model.output) is list else [model.output]

        y_true_layers = self.construct_y_true_layers()

        # Define loss layer
        loss_inputs = [*original_outputs, *y_true_layers]
        loss_args = {'anchors': self.config.anchors, 'num_classes': self.config.num_classes(),
                     'ignore_thresh': self.config.ignore_thresh}
        loss_layer = keras.layers.Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                                         arguments=loss_args)(loss_inputs)

        # Wrap the model by adding the y_true_layers to the input and setting the output to the loss layer
        model = keras.models.Model([*original_inputs, *y_true_layers], loss_layer)
        return model

    def training_mode(self):
        """Set the current mode to TRAINING."""
        self.set_mode(Mode.TRAINING)

    def inference_mode(self):
        """Set the current mode to STANDARD (i.e. for inference)."""
        self.set_mode(Mode.STANDARD)

    def data_generator(self, annotations, image_directory, batch_size, to_fit=True, random=True, jittering_params=None):
        """
        Create a data generator based on the given data.

        Parameters
        ----------
        annotations : dict
            The annotations for the given image data.
        image_directory : path-like
            Path to the directory to load the image data from.
        batch_size : int
            The batch size.
        to_fit : boolean
            Whether the generator is used to fit a model.
        random : boolean
            If no jittering_params are defined and random is False, the jittering probabilities are set to 0 (so no
            random data augmentations are performed. Otherwise, the default values defined in DataGenerator are used.
        jittering_params : dict
            Dict describing the jittering parameters rotation_probability, jittering_probability, rotation_angles,
            and jittering_range. Default depends on the random parameter (see above).

        Returns
        -------
        generator.DataGenerator
            The data generator.
        """
        if jittering_params is None:
            jittering_params = {} if random else {"rotation_probability": 0, "jittering_probability": 0}

        return src.generator.DataGenerator(batch_size, self.config, image_directory, annotations_dict=annotations,
                                           to_fit=to_fit, shuffle=True, shuffle_annotations=True, max_boxes=20,
                                           **jittering_params)

    @staticmethod
    def compile_model(model, learning_rate):
        """
        Compile the given model with an Adam optimizer with the specified learning rate.

        Parameters
        ----------
        model : keras.models.Model
            The model to compile
        learning_rate : double
            The learning rate to use for the Adam optimizer.
        """
        loss = {'yolo_loss': lambda y_true, y_prediction: y_prediction}
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=optimizer, loss=loss)

    @staticmethod
    def train_model_wrapper(model, generators, data_len, batch_size, learning_rate, epochs, initial_epoch=0,
                            callbacks=None, **kwargs):
        """
        Helper method to perform a training step on the given model and generator with the specified parameters.
        The given model is compiled before starting the training.

        Parameters
        ----------
        model : keras.models.Model
            The model to train.
        generators : tuple(generator.DataGenerator)
            A pair of generators for the training/validation set.
        data_len : tuple(int)
            The number of images in the training/validation set.
        batch_size : int
            The batch-size to use for training.
        learning_rate : double
            The learning rate to use for the optimizer.
        epochs : int
            The number of epochs to train.
        initial_epoch : int
            The initial epoch (used when performing multiple training steps). (Default: 0)
        callbacks : List
            Optional list of keras.callbacks.Callback to apply during training.

        Returns
        -------
        history : History
            Returns the training history (return value of keras fit_generator).
        """
        if callbacks is None:
            callbacks = []
        TinyYoloV3.compile_model(model, learning_rate)

        steps_per_epoch = [max(1, length // batch_size) for length in data_len]
        return model.fit_generator(generators[0], steps_per_epoch=steps_per_epoch[0], validation_data=generators[1],
                                   validation_steps=steps_per_epoch[1], epochs=epochs, initial_epoch=initial_epoch,
                                   callbacks=callbacks, **kwargs)

    def train(self, train_annotations, val_annotations, image_directory, learning_rate, batch_size, epochs,
              initial_epoch=0, callbacks=None, out_path=None, jittering_params=None, random=True, **kwargs):
        """
        Train this model on the given data with the specified parameters.

        Parameters
        ----------
        train_annotations : dict
            The annotations for the training set.
        val_annotations : dict
            The annotations for the validation set.
        image_directory : path-like
            The directory to load the images from.
        learning_rate : double
            The learning rate to use for the optimizer.
        batch_size : int
            The batch-size to use for training.
        epochs : int
            The number of epochs to train.
        initial_epoch : int
            The initial epoch (used when performing multiple training steps). (Default: 0)
        callbacks : List
            Optional list of keras.callbacks.Callback to apply during training.
        out_path : Optional path-like
            If specified, the trained model weights are saved to this path.
        jittering_params : dict
            Dict describing the jittering parameters rotation_probability, jittering_probability, rotation_angles,
            and jittering_range. Default depends on the random parameter (see above).
        random : boolean
            If no jittering_params are defined and random is False, the jittering probabilities are set to 0 (so no
            random data augmentations are performed. Otherwise, the default values defined in DataGenerator are used.
            This applies only to the training data, the validation data is never augmented.

        Returns
        -------
        history : History
            Returns the training history (return value of keras fit_generator).
        """
        print(f"Train for {epochs - initial_epoch} epochs on {len(train_annotations)} samples with "
              f"{len(val_annotations)} validation samples and batch size {batch_size}.")

        data_generator_train = self.data_generator(train_annotations, image_directory, batch_size,
                                                   jittering_params=jittering_params, random=random)
        data_generator_validation = self.data_generator(val_annotations, image_directory, batch_size, random=False)
        generators = [data_generator_train, data_generator_validation]
        data_len = [len(train_annotations), len(val_annotations)]

        history = TinyYoloV3.train_model_wrapper(self.model, generators, data_len, batch_size, learning_rate, epochs,
                                                 initial_epoch, callbacks, **kwargs)
        self.compiled = True

        if out_path is not None:
            self.save_weights(out_path)

        return history

    def evaluate(self, annotations, image_directory, batch_size=1):
        """
        Evaluate the current model on the given data with the specified batch size.

        Parameters
        ----------
        annotations : dict
            The annotations for the given image data.
        image_directory : path-like
            Path to the directory to load the image data from.
        batch_size : int
            The batch size to use for the evaluation.

        Returns
        -------
        loss : double
            The loss on the given data.
        """
        self.training_mode()
        if not self.compiled:
            TinyYoloV3.compile_model(self.model, 0)
            self.compiled = True
        data_generator = self.data_generator(annotations, image_directory, batch_size, random=False)
        return self.model.evaluate_generator(data_generator)

    def save_weights(self, path):
        """
        Save the model weights at the given path.

        Parameters
        ----------
        path : path-like
            The path to save the weights at.
        """
        self.model.save_weights(path)

    def load_weights(self, path):
        """
        Load the model weights from the given path.

        Parameters
        ----------
        path : path-like
            The path to load the weights from.
        """
        self.model.load_weights(path, by_name=False)

    @staticmethod
    def get_y_shapes(model_config):
        """
        Determine the shape of the y_true input layers for a given configuration.
        There is one shape per output layer. Each output layer has a different scale/grid shape.
        The y_true shape is defined as: grid_height X grid_width X #anchors X output_size
        with output_size = #classes + 5.

        Parameters
        ----------
        model_config : YoloConfig
            The model configuration.

        Returns
        -------
        y_shapes : List(tuple)
            A list of the y_shapes as defined above. One shape per output layer.
        """
        width, height = model_config.input_size

        anchors_per_scale = len(model_config.anchors) // len(model_config.scales)
        output_size = model_config.num_classes() + 5

        grid_shapes = [(height // scale, width // scale) for _, scale in model_config.scales.items()]
        y_shapes = [(*shape, anchors_per_scale, output_size) for shape in grid_shapes]

        return y_shapes

    @staticmethod
    def make_model(config, num_classes=None):
        """
        Create a new Tiny-YOLO v3 model from the given configuration.

        Parameters
        ----------
        config : YoloConfig
            The model configuration.
        num_classes: int
            Used to load trained on different number of classes

        Returns
        -------
        yolov3 : keras.models.Model
            The created model.
        """
        # extract number of classes
        if num_classes is None:
            num_classes = config.num_classes()

        # original yolov3 parameters
        # Layer0
        input_shape = (config.input_size.height, config.input_size.width, config.channels)
        input_layer = Input(shape=input_shape)
        layer0 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=False)(input_layer)
        layer0 = BatchNormalization()(layer0)
        layer0 = LeakyReLU(alpha=0.1)(layer0)
        # Layer 1
        layer1 = MaxPooling2D(pool_size=(2, 2))(layer0)
        # Layer 2
        layer2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False)(layer1)
        layer2 = BatchNormalization()(layer2)
        layer2 = LeakyReLU(alpha=0.1)(layer2)
        # Layer 3
        layer3 = MaxPooling2D(pool_size=(2, 2))(layer2)
        # Layer 4
        layer4 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False)(layer3)
        layer4 = BatchNormalization()(layer4)
        layer4 = LeakyReLU(alpha=0.1)(layer4)
        # Layer 5
        layer5 = MaxPooling2D(pool_size=(2, 2))(layer4)
        # Layer 6
        layer6 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False)(layer5)
        layer6 = BatchNormalization()(layer6)
        layer6 = LeakyReLU(alpha=0.1)(layer6)
        # Layer 7
        layer7 = MaxPooling2D(pool_size=(2, 2))(layer6)
        # Layer 8
        layer8 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False)(layer7)
        layer8 = BatchNormalization()(layer8)
        layer8 = LeakyReLU(alpha=0.1)(layer8)
        # Layer 9
        layer9 = MaxPooling2D(pool_size=(2, 2))(layer8)
        # Layer 10
        layer10 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False)(layer9)
        layer10 = BatchNormalization()(layer10)
        layer10 = LeakyReLU(alpha=0.1)(layer10)
        # Layer 11
        layer11 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(layer10)
        # Layer 12
        layer12 = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False)(layer11)
        layer12 = BatchNormalization()(layer12)
        layer12 = LeakyReLU(alpha=0.1)(layer12)
        # Layer 13
        layer13 = Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False)(layer12)
        layer13 = BatchNormalization()(layer13)
        layer13 = LeakyReLU(alpha=0.1)(layer13)
        # Layer 14
        layer14 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False)(layer13)
        layer14 = BatchNormalization()(layer14)
        layer14 = LeakyReLU(alpha=0.1)(layer14)
        # Layer 15 / YOLO
        layer15 = Conv2D(config.num_anchors() * (num_classes + 5), (1, 1),
                         strides=(1, 1), padding='same', use_bias=True)(layer14)
        layer15 = Activation('linear')(layer15)
        # Layer 16 Yolo (see layer 15)
        # Layer 17 Route
        # Layer 18
        layer18 = Conv2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False)(layer13)
        layer18 = BatchNormalization()(layer18)
        layer18 = LeakyReLU(alpha=0.1)(layer18)
        # Layer 19
        layer19 = UpSampling2D((2, 2))(layer18)
        # Layer 20 Route
        layer20 = Concatenate()([layer19, layer8])
        # Layer 21
        layer21 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False)(layer20)
        layer21 = BatchNormalization()(layer21)
        layer21 = LeakyReLU(alpha=0.1)(layer21)
        # Layer 22 / YOLO
        layer22 = Conv2D(config.num_anchors() * (num_classes + 5), (1, 1),
                         strides=(1, 1), padding='same', use_bias=True)(layer21)
        layer22 = Activation('linear')(layer22)
        # Layer 23 YOLO
        # Finalize Model
        yolov3 = Model(input_layer, [layer15, layer22])

        return yolov3

    def freeze(self, layer_ids=None):
        """
        Freeze the specified layers (or all, if no layers are specified).

        Parameters
        ----------
        layer_ids : List[int]
            The layers to freeze, if None, all layers are frozen.
        """
        if layer_ids is None:
            layer_ids = range(len(self.model.layers))

        for i in layer_ids:
            self.model.layers[i].trainable = False

    def freeze_all_but_output(self):
        """Freeze all layers expect the output layer."""
        freeze_count = len(self.model.layers) - self.output_layers * 2
        self.freeze(range(freeze_count))

    def freeze_darknet_body(self):
        """Freeze all layers in the darknet body."""
        self.freeze(range(self.darknet_body_layers))

    def unfreeze(self):
        """Unfreeze all layers."""
        for layer in self.model.layers:
            layer.trainable = True

    def replace_output_layers(self, classes=None):
        """
        Replace the output layers with new layers. The new shape is defined by the number of classes given (or those
        specified in the config).

        Parameters
        ----------
        classes : List
            A list of class names.
        """
        if classes is not None:
            self.config.class_names = classes
        num_classes = self.config.num_classes()

        # Pop top 4 layers
        for i in range(4):
            self.model.layers.pop()

        # Add new output layers
        prev1 = self.model.layers[-2]
        prev2 = self.model.layers[-1]

        yolo_output_shape = self.config.num_anchors() * (num_classes + 5)
        yolo1 = Conv2D(filters=yolo_output_shape, kernel_size=(1, 1), strides=(1, 1), padding='same',
                       use_bias=True)(prev1.output)
        yolo1 = Activation('linear')(yolo1)

        yolo2 = Conv2D(filters=yolo_output_shape, kernel_size=(1, 1), strides=(1, 1), padding='same',
                       use_bias=True)(prev2.output)
        yolo2 = Activation('linear')(yolo2)

        self.model = Model(inputs=self.model.input, outputs=[yolo1, yolo2])
        self.mode = Mode.STANDARD
        self.update_original_in_and_output_layers()

    def from_tensorRT(self, session):
        """
        Initializes the model for inference from a tensor RT graph. Expects that the RT graph is already set up in the
        given session.

        Parameters
        ----------
        session : Keras session
            The session to use for prediction.
        """
        self.session = session
        output_tensors = [self.session.graph.get_tensor_by_name(f'activation_{i}/Identity:0') for i in range(3, 5)]
        input_tensors = self.session.graph.get_tensor_by_name('input_1:0')

        self.original_output = output_tensors
        self.original_input = input_tensors

        self.set_mode(Mode.STANDARD)
        _eval = yolo_eval(output_tensors, self.config.anchors,
                          self.config.num_classes(), self.input_image_shape,
                          score_threshold=self.config.score, iou_threshold=self.config.iou)
        self.boxes, self.scores, self.classes = _eval
