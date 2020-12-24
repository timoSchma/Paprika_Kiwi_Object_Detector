import keras
import numpy as np

from src.generator import DataGenerator, preprocess_true_boxes
from src.annotated_image import AnnotatedImage
from src.utils import split_list


class BottleneckFeatures:
    """
    This class represents bottleneck features, i.e. features computed by the bottleneck model (the model without the
    last few layers).
    """

    def __init__(self, bottleneck_features=None, image_keys=None, annotations_dict=None):
        """
        Create a new bottleneck features object.

        Parameters
        ----------
        bottleneck_features : np.array
            The bottleneck features.
        image_keys : List
            A list of the image names assigned to this dataset. Used to filter the annotations dict-
        annotations_dict : dict
            The annotations for the images in this dataset. May contain more images than assigned to this dataset.
            Only the images in image_keys are used.
        """
        self.features = bottleneck_features
        self.image_keys = image_keys
        self.annotations_dict = annotations_dict

    def train_test_split(self, training_indices, validation_indices):
        """
        Split the data into training and validation set based on the given indices.

        Parameters
        ----------
        training_indices : List[int]
            The indices of the training images.
        validation_indices : List[int]
            The indices of the validation images.

        Returns
        -------
        train : BottleneckFeatures
            The training set.
        val : BottleneckFeatures
            The validation set.
        """
        train_features = [features[training_indices] for features in self.features]
        val_features = [features[validation_indices] for features in self.features]
        train_keys, val_keys = split_list(self.image_keys, [training_indices, validation_indices])

        train = BottleneckFeatures(train_features, train_keys, self.annotations_dict)
        val = BottleneckFeatures(val_features, val_keys, self.annotations_dict)
        return train, val

    def create_from_data(self, annotations_dict, image_directory, bottleneck_model, yolo_config):
        """
        Creates the bottleneck features from the data set using the given bottleneck model.

        Parameters
        ----------
        annotations_dict : dict
            The annotations describing the data labels.
        image_directory : path-like
            The path to load the images from.
        bottleneck_model : keras.models.Model
            The bottleneck model used to compute the bottleneck features for a given image.
        yolo_config : YoloConfig
            The YOLO configuration.
        """
        batch_size = 1
        generator = DataGenerator(batch_size, yolo_config, image_directory, annotations_dict=annotations_dict,
                                  to_fit=True, shuffle=False, shuffle_annotations=True, max_boxes=20,
                                  rotation_probability=0, jittering_probability=0)
        self.image_keys = list(annotations_dict.keys())

        self.features = bottleneck_model.predict_generator(generator)
        self.annotations_dict = annotations_dict

    def save(self, path):
        """
        Save the bottleneck features at the given path.

        Parameters
        ----------
        path : path-like
            The path to save the features at.
        """
        feature_dict = {f"bottleneck_features_{i}": features for i, features in enumerate(self.features)}
        np.savez(path, **feature_dict)

    def load(self, path):
        """
        Load the bottleneck features from the given path.

        Parameters
        ----------
        path : path-like
            Path to load the features from.
        """
        feature_dict = np.load(path)
        self.features = list(feature_dict.values())

    def generator(self, batch_size, yolo_config, to_fit=True, shuffle=True, shuffle_annotations=True, max_boxes=20):
        """
        Create a BottleneckGenerator based on these features.

        Parameters
        ----------
        batch_size : int
            The batch size.
        yolo_config : YoloConfig
            The YOLO configuration.
        to_fit : boolean
            Whether this generator is used to fit a model (i.e. the labels should be returned) or just used for
            inference (i.e. no labels required). (default: True)
        shuffle : boolean
            Whether the data should be shuffled after each epoch. (default: True)
        shuffle_annotations : boolean
            Whether the annotations of a single image should be shuffled. (default: True)
        max_boxes : int
            The maximum number of boxes per image (all further boxes are truncated, default is 20).

        Returns
        -------
        BottleneckGenerator
            The created generator.
        """
        return BottleneckGenerator(batch_size, self.annotations_dict, self, yolo_config, to_fit, shuffle,
                                   shuffle_annotations, max_boxes)


class BottleneckGenerator(keras.utils.Sequence):
    """A data generator for bottleneck features based on the keras sequence."""

    def __init__(self, batch_size, annotations_dict, bottleneck_features, model_config, to_fit=True, shuffle=True,
                 shuffle_annotations=True, max_boxes=20):
        """
        Create a new generator instance over the given bottleneck features.

        Parameters
        ----------
        batch_size : int
            The batch size (number of images per batch)
        annotations_dict : dict
            The annotations describing the data labels.
        bottleneck_features : BottleneckFeatures
            The set of bottleneck features to iterate over in each epoch.
        model_config : YoloConfig
            The YOLO configuration.
        to_fit : boolean
            Whether this generator is used to fit a model (i.e. the labels should be returned) or just used for
            inference (i.e. no labels required). (default: True)
        shuffle : boolean
            Whether the data should be shuffled after each epoch. (default: True)
        shuffle_annotations : boolean
            Whether the annotations of a single image should be shuffled. (default: True)
        max_boxes : int
            The maximum number of boxes per image (all further boxes are truncated, default is 20).
        """
        self.model_config = model_config

        self.to_fit = to_fit
        self.shuffle = shuffle
        self.shuffle_annotations = shuffle_annotations
        self.max_boxes = max_boxes

        self.bottleneck_features = bottleneck_features
        self.annotations_dict = annotations_dict

        self.n = len(self.bottleneck_features.features[0])
        self.batch_size = min(batch_size, self.n)
        self.indices = list(range(self.n))

        self.on_epoch_end()

    def __len__(self):
        """
        Returns the length, i.e. the number of batches per epoch.

        Returns
        -------
        int
            The number of batches per epoch.
        """
        return self.n // self.batch_size

    def __getitem__(self, index):
        """
        Get the batch at the given index. Depending on self.to_fit, either data and labels (x and y) or only the
        data (x) is returned.

        Parameters
        ----------
        index : int
            The index of the requested batch.

        Returns
        -------
        item : List of (annotated) features, if to_fit is True: additionally np.array of zeros
            The requested item (= (annotated) features of the requested batch). See generate_annotated_batch_features
            and generate_batch_features for more details.
        """
        current_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        if self.to_fit:
            item = self.generate_annotated_batch_features(current_indices)
        else:
            item = self.generate_batch_features(current_indices)
        return item

    def generate_batch_features(self, current_indices):
        """
        Generates a batch of features from the given image indices.

        Parameters
        ----------
        current_indices : List[int]
            A list of image indices specifying which images are in this batch.

        Returns
        -------
        features : List of features
            The features of the requested batch of images.
        """
        return [features[current_indices] for features in self.bottleneck_features.features]

    def generate_annotated_batch_features(self, current_indices):
        """
        Generates a batch of features and labels from the given image indices.

        Parameters
        ----------
        current_indices : List[int]
            A list of image indices specifying which images are in this batch.

        Returns
        -------
        annotated_features : List of features and y_true
            A list of the annotated features (features and annotation y_true) for each image in the batch.
        zeros : np.array
            An array of 'batch_size' zeros.
        """
        current_features = self.generate_batch_features(current_indices)

        # Create annotations for all images in the current batch
        file_names = [self.bottleneck_features.image_keys[i] for i in current_indices]
        annotations = [AnnotatedImage() for _ in file_names]
        for annotation, file_name in zip(annotations, file_names):
            annotation.annotations_from_dict(self.annotations_dict, file_name)

        # Convert annotations to yolo format
        box_data = np.array([annotation.get_annotations_in_yolo_format(
            self.model_config.class_names, shuffle=self.shuffle_annotations, max_boxes=self.max_boxes)
            for annotation in annotations])
        y_true = preprocess_true_boxes(box_data, self.model_config)

        return [*current_features, *y_true], np.zeros(self.batch_size)

    def on_epoch_end(self):
        """Shuffles the keys after each epoch (if self.shuffle is set)."""
        if self.shuffle:
            np.random.shuffle(self.indices)
