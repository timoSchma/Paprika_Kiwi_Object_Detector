import numpy as np
import pathlib
import keras
import random
import json

import src.model
from src.annotated_image import AnnotatedImage, Image
from src.preprocessing import color_jittering


class DataGenerator(keras.utils.Sequence):
    """A data generator for images with object detection annotations based on the keras sequence."""

    def __init__(self, batch_size, model_config, image_directory, image_pattern='*.jpg',
                 annotations_dict=None, annotations_path=None, to_fit=True, shuffle=True, shuffle_annotations=True,
                 max_boxes=20, rotation_probability=0.5, jittering_probability=0.5, rotation_angles=None,
                 jittering_range=None, verify_data=True, skip_invalid_images=False):
        """
        Create a new generator for the given data with the specified parameters.

        Parameters
        ----------
        batch_size : int
            The batch size (number of images per batch)
        model_config : YoloConfig
            The YOLO configuration.
        image_directory : path-like
            The directory to load the images from.
        image_pattern :
            A regex-pattern to filter the files at the given image directory. Default: '*.jpg' to retrieve all files
            with the file extension '*.jpg'.
        annotations_dict : dict
            The annotations describing the data labels.
        annotations_path : path-like
            Alternative to annotations_dict: load the dict from the given path.
        to_fit : boolean
            Whether this generator is used to fit a model (i.e. the labels should be returned) or just used for
            inference (i.e. no labels required). (default: True)
        shuffle : boolean
            Whether the data should be shuffled after each epoch. (default: True)
        shuffle_annotations : boolean
            Whether the annotations of a single image should be shuffled. (default: True)
        max_boxes : int
            The maximum number of boxes per image (all further boxes are truncated, default is 20).
        rotation_probability : double
            The probability for data augmentation with rotation. (Default: 0.5)
        jittering_probability : double
            The probability for data augmentation with jittering. (Default: 0.5)
        rotation_angles : List[int]
            A list of angles from [0, 360] to randomly rotate the image with. In general, multiples of 90 degree are
            recommended for annotated images as other angles significantly increase the bounding box size.
            (Default: [90, 180, 270])
        jittering_range : dict
            A dict defining the jittering ranges for the three values 'hue', 'sat', and 'val' for color jittering. It is
            also possible to overwrite only some of the parameters.
            (Default: {'hue': 0.1, 'sat': 1.5, 'val': 1.5})
        verify_data : boolean
            Whether the given input data should be verified when creating this generator (e.g. making sure all required
            parameters are given, all images can be loaded, etc.). (Default: True)
        skip_invalid_images : boolean
            Whether to skip invalid images found during the verification. (Default: True)
        """
        self.model_config = model_config

        self.image_directory = image_directory
        self.to_fit = to_fit
        self.shuffle = shuffle
        self.shuffle_annotations = shuffle_annotations
        self.max_boxes = max_boxes

        self.rotation_probability = rotation_probability
        self.rotation_angles = [90, 180, 270] if rotation_angles is None else rotation_angles

        self.jittering_probability = jittering_probability
        self.jittering_range = {"hue": 0.1, "sat": 1.5, "val": 1.5}
        if jittering_range is not None:
            self.jittering_range = {**self.jittering_range, **jittering_range}

        self.image_paths = None
        self.annotations_dict = None

        if to_fit:  # training case -> load and use annotations
            if annotations_dict is None:
                with open(annotations_path, 'r') as json_file:
                    annotations_dict = json.loads(json_file.read())
            self.annotations_dict = annotations_dict
            self.image_paths = [pathlib.Path(self.image_directory, file_name)
                                for file_name in self.annotations_dict.keys()]
        else:  # inference case -> load only images, no annotations needed
            self.image_paths = [file for file in pathlib.Path(self.image_directory).glob(image_pattern)]

        self.n = len(self.image_paths)
        self.batch_size = min(batch_size, self.n)
        self.on_epoch_end()

        if verify_data:
            self._verify_data(skip_invalid_images)

    def _verify_data(self, skip_invalid_images):
        """
        Helper function to verify the input data.
        Ensures that:
        - the parameters image_paths and annotations_dict are set (annotations_dict only if to_fit is True)
        - for each image: the image can be loaded as Image/AnnotatedImage with annotation if to_fit is True
        All errors are printed, if skip_invalid_images is True, invalid images are skipped.

        Parameters
        ----------
        skip_invalid_images : boolean
            Whether to skip invalid images in the data generator, i.e. removing them from the generator's dataset.
        """
        if self.image_paths is None:
            raise ValueError("DataGenerator: image_paths not set.")
        if self.to_fit and self.annotations_dict is None:
            raise ValueError("DataGenerator: to_fit but annotations_dict not set.")

        errors = []
        for path in self.image_paths:
            try:
                if self.to_fit:
                    AnnotatedImage(image_path=path, annotation_dict=self.annotations_dict)
                else:
                    Image(path=path)
            except Exception as error:
                errors.append((path, error))

        if errors:
            error_messages = '\n- '.join(f"Error {error} for image {str(path)}" for path, error in errors)
            if not skip_invalid_images:
                raise ValueError(f"Could not load images:\n- {error_messages}")
            else:
                print(f"Error while loading some images, these will be skipped:\n- {error_messages}")
                skipped_paths = [error[0] for error in errors]
                self.image_paths = [path for path in self.image_paths if path not in skipped_paths]

    def __len__(self):
        """
        Returns the length, i.e. the number of batches per epoch.

        Returns
        -------
        int
            The number of batches per epoch.
        """
        if self.batch_size <= 0:
            return 0
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
        item : List of (annotated) images, if to_fit is True: additionally np.array of zeros
            The requested item (= (annotated) features of the requested batch). See generate_annotated_image_data
            and generate_image_data for more details.

        """
        current_files = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]

        if self.to_fit:
            return self.generate_annotated_image_data(current_files)
        else:
            return self.generate_image_data(current_files)

    def random_rotation(self, image):
        """
        Randomly rotate the given image by an angle randomly chosen from self.rotation_angles.
        The image is only rotated with a probability of self.rotation_probability.

        Parameters
        ----------
        image : Image or AnnotatedImage
            The image to rotate.

        Returns
        -------
        image : Image or AnnotatedImage
            The rotated image.
        """
        if self.rotation_probability is None or random.random() > self.rotation_probability:
            return image
        angle = random.choice(self.rotation_angles)
        image.rotate(angle)
        return image

    def random_color_jittering(self, image_data):
        """
        Randomly adjust color and brightness by the ranges defined in self.jittering_range.
        The image is only jittered with a probability of self.jittering_probability.

        Parameters
        ----------
        image_data : np.array
            The image data of the image to perform color jittering on.

        Returns
        -------
        np.array
            The image data after color jittering.
        """
        if self.jittering_probability is None or random.random() > self.jittering_probability:
            return image_data
        return color_jittering(image_data, **self.jittering_range)

    def generate_annotated_image_data(self, current_files):
        """
        Loads and prepares images with annotations (e.g. for training)

        Parameters
        ----------
        current_files : List[string]
            A list of image files specifying which images are in this batch.

        Returns
        -------
        annotated_data : List of two np.arrays
            A list of the images and annotations y_true for each image in the batch.
        zeros : np.array
            An array of 'batch_size' zeros.
        """
        def prepare_annotated_image(path):
            # load image and annotations
            annotated_image = AnnotatedImage(image_path=path, annotation_dict=self.annotations_dict)
            # perform data augmentations
            annotated_image = self.random_rotation(annotated_image)
            image_data = annotated_image.image.image_data / 255
            image_data = self.random_color_jittering(image_data)
            annotated_image.image.image_data = image_data
            return annotated_image

        def prepare_annotation(annotated_image):
            return annotated_image.get_annotations_in_yolo_format(self.model_config.class_names,
                                                                  shuffle=self.shuffle_annotations,
                                                                  max_boxes=self.max_boxes)

        annotated_images = [prepare_annotated_image(path) for path in current_files]

        img_data = np.array([annotated_image.image.image_data for annotated_image in annotated_images])
        box_data = np.array([prepare_annotation(annotated_image) for annotated_image in annotated_images])

        y_true = preprocess_true_boxes(box_data, self.model_config)

        return [img_data, *y_true], np.zeros(self.batch_size)

    def generate_image_data(self, current_files):
        """
        Loads and prepares images without annotations (e.g. for inference).

        Parameters
        ----------
        current_files : List[string]
            A list of image files specifying which images are in this batch.

        Returns
        -------
        np.array
            The image data in the requested batch.
        """

        def prepare_image(path):
            image = Image(path=path)
            image = self.random_rotation(image)
            image_data = image.image_data / 255
            image_data = self.random_color_jittering(image_data)
            return image_data

        images = [prepare_image(path) for path in current_files]
        return np.array(images)

    def on_epoch_end(self):
        """Shuffles the keys after each epoch (if self.shuffle is set)."""
        if self.shuffle:
            np.random.shuffle(self.image_paths)


def preprocess_true_boxes(true_boxes, model_config):
    """
    Helper function to convert the bounding box annotations for a batch of images into the format expected by YOLO.

    Parameters
    ----------
    true_boxes : np.array
        An array containing all bounding boxes for all images in the batch. Each box is described by 5 values:
        4 coordinates describing the box (center-x, center-y, width, height) and one class id. The coordinates are
        given as absolute values.
    model_config : YoloConfig
        The YOLO configuration.

    Returns
    -------
    y_true : list of np.array
        The ground truth for y to write the encoded annotations to. One np.array per output layer.
    """
    # Prepare and validate parameters
    true_boxes = np.array(true_boxes, dtype='float32')
    image_count = true_boxes.shape[0]  # number of images

    input_shape = np.array(model_config.input_size[::-1], dtype='int32')

    if np.array(true_boxes[..., 4] >= model_config.num_classes()).any():
        raise ValueError("class id must be less than num_classes")

    anchor_mask = [[3, 4, 5], [0, 1, 2]]

    # get bounding box sizes (width, height for each box)
    boxes_size_wh = true_boxes[..., 2:4]

    # Normalize to relative measures by dividing the position and size by the image size
    # use input_shape[::-1] to invert the shape to width, height order
    relative_boxes = np.zeros_like(true_boxes)
    relative_boxes[..., 0:2] = true_boxes[..., 0:2] / input_shape[::-1]
    relative_boxes[..., 2:4] = true_boxes[..., 2:4] / input_shape[::-1]
    relative_boxes[..., 4] = true_boxes[..., 4]

    # Compute the grid shape for each output layer
    height, width = input_shape[0:2]
    grid_shapes = [(height // scale, width // scale) for scale in model_config.scales.values()]

    # Construct y-input array, initialize with all zeros
    y_shapes = src.model.TinyYoloV3.get_y_shapes(model_config)
    y_true = [np.zeros((image_count, *shape), dtype='float32') for shape in y_shapes]

    # Set mask to exclude zero-width boxes
    valid_mask = boxes_size_wh[..., 0] > 0

    # For each image: set the annotations in y_true
    for image_id in range(image_count):
        # Get boxes for this image and discard zero-width boxes
        current_boxes_wh = boxes_size_wh[image_id, valid_mask[image_id]]

        # Skip image if no non-zero boxes are left
        if len(current_boxes_wh) == 0:
            continue

        # Encode the remaining boxes
        set_y_true(y_true, image_id, model_config, anchor_mask, current_boxes_wh, relative_boxes, grid_shapes)

    return y_true


def set_y_true(y_true, image_id, model_config, anchor_mask, boxes_wh, relative_boxes, grid_shapes):
    """
    Prepare y_true for a single image by encoding all bounding boxes.
    Process to encode a box:
    - determine the best fitting anchor box
    - encode the box in the output size 5 + num_classes
    - determine the corresponding layer and corresponding anchor id
    - determine the responsible grid cell in that layer
    - write the encoded annotation to the correct position in y_true (defined by the layer, image id, grid cell and
      anchor id)

    Parameters
    ----------
    y_true : list of np.array
        The ground truth for y to write the encoded annotations to. One np.array per output layer.
    image_id : int
        The id of the image currently processed.
    model_config : YoloConfig
        The YOLO configuration.
    anchor_mask : List[List[int]]
        A list containing for each output layer a list of anchors.
    boxes_wh : np.array
        The absolute size (width, height) of the bounding boxes on this image.
    relative_boxes : np.array
        An array defining all bounding boxes with 5 entries per box: 4 coordinates describing the box (width, height,
        center-x, center-y) and one class id. The coordinates are given relative to the image size.
    grid_shapes : List
        A list of shapes defining the grid on each output layer, i.e. the number of grid cells in each direction.
    """
    # Find best anchor for each bounding box
    best_anchors = find_best_anchors(boxes_wh, model_config.anchors)

    # Encode each bounding box and write it into the correct position in y_true
    for bounding_box_id, best_anchor in enumerate(best_anchors):
        # Encode the bounding box
        current_box = relative_boxes[image_id, bounding_box_id]
        encoded_annotation = encode_annotation(model_config.num_classes(), current_box)

        # Determine the layer and anchor id and the responsible grid cell
        layer_id, anchor_id = determine_layer_and_anchor_id(anchor_mask, best_anchor)
        grid_y, grid_x = determine_responsible_grid_cell(current_box, grid_shapes, layer_id)

        # Write the encoded annotation to the correct position in y_true
        y_true[layer_id][image_id, grid_x, grid_y, anchor_id] = encoded_annotation


def determine_responsible_grid_cell(box, grid_shapes, layer_id):
    """
    Determines the grid cell (x and y coordinate) in the given layer responsible for the given bounding box.
    The responsible grid cell is that cell containing the center of the box.

    Parameters
    ----------
    box : np.array
        An array defining the bounding box with box[0] == center-y and box[1] == center-x. The values are given
        relative to the image size.
    grid_shapes : List
        A list of shapes defining the grid on each output layer, i.e. the number of grid cells in each direction.
    layer_id : int
        The id of the output layer responsible for the box.

    Returns
    -------
    grid_y : int
        The y-coordinate of the responsible grid cell.
    grid_x : int
        The x-coordinate of the responsible grid cell.
    """
    grid_y, grid_x = (box[0:2] * grid_shapes[layer_id]).astype(int)
    return grid_y, grid_x


def determine_layer_and_anchor_id(anchor_mask, best_anchor):
    """
    Determine the correct output layer for a selected best anchor.

    Parameters
    ----------
    anchor_mask : List[List[int]]
        A list containing for each output layer a list of anchors.
    best_anchor : int
        The best anchor selected from those in the anchor mask.

    Returns
    -------
    layer_id : int
        The id of the responsible output layer.
    anchor_id : int
        The id of the anchor withing the output layer.
    """
    layer_id = [layer_id for layer_id, mask in enumerate(anchor_mask) if best_anchor in mask][0]
    anchor_id = anchor_mask[layer_id].index(best_anchor)
    return layer_id, anchor_id


def encode_annotation(num_classes, box):
    """
    Encode an annotation (given by the np.array box with 5 entries: 4-coordinates and the class id) in an np.array of
    size 5 + num_classes (the class is one-hot encoded).

    Parameters
    ----------
    num_classes : int
        The number of classes.
    box : np.array
        An array with 5 entries: 4 coordinates describing the box (width, height, center-x, center-y) and one class id.
        The coordinates are given relative to the image size.

    Returns
    -------
    encoded_annotation : np.array
        An array of size 5 + num_classes. Entries 0:4 are the coordinates, 4 is the confidence score (1) and 5: is the
        one-hot encoded class id.
    """
    # Encode box position & size and object confidence score (=1 since it's a training image)
    encoded_annotation = np.zeros(5 + num_classes)
    encoded_annotation[0:5] = list(box[0:4]) + [1]

    # One-hot encode the class in [5:]
    class_id = int(box[4])
    encoded_annotation[5 + class_id] = 1

    return encoded_annotation


def compute_areas(width_height_array):
    """
    Computes the area for each box in the given array of boxes by multiplying each width with the corresponding height.

    Parameters
    ----------
    width_height_array : np.array
        An array of boxes defined by their width [..., 0] and height [..., 1].

    Returns
    -------
    np.array
        An array of box areas.
    """
    return width_height_array[..., 0] * width_height_array[..., 1]


def get_min_max_corners(width_height_array):
    """
    Computes the top-left (min) and bottom-right (max) corner of an array of boxes defined by their width and height.
    The computed corners are relative to the box center (i.e. the box center is at 0,0).

    Parameters
    ----------
    width_height_array : np.array
        An array of boxes defined by their width [..., 0] and height [..., 1].

    Returns
    -------
    top_left : np.array
        An array containing the top-left corner for each box (x and y are <=0, since the box center is at 0,0)
    bottom_right : np.array
        An array containing the bottom-right corner for each box (x and y are >=0, since the box center is at 0,0)
    """
    bottom_right = width_height_array / 2.
    top_left = -bottom_right
    return top_left, bottom_right


def intersect(box1, box2):
    """
    Compute the intersection of two boxes (which is a new box). Boxes are defined by the two corners top_left and
    bottom_right.

    Parameters
    ----------
    box1 : np.array
        The first box in the format: [[top, left], [bottom, right]]
    box2 : np.array
        The second box in the format: [[top, left], [bottom, right]]

    Returns
    -------
    top_left : np.array
        The top-left corner of the intersection in the format: [top, left]
    bottom_right : np.array
        The bottom-right corner of the intersection in the format: [bottom, right]
    """
    top_left = np.maximum(box1[0], box2[0])
    bottom_right = np.minimum(box1[1], box2[1])
    return top_left, bottom_right


def find_best_anchors(boxes_width_height, anchors):
    """
    For an array of bounding boxes (defined by their width and height), find the best anchor box using the maximum IoU.
    This function computes the intersection between each input box and each anchor box, the area of all these boxes and
    finally the IoUs. Then, the anchor box with maximum IoU is selected and returned for each of the input boxes.

    Parameters
    ----------
    boxes_width_height : np.array
        An array describing all input bounding boxes, for which this function shall determine the best anchor.
        Each box is defined by its width and height.
    anchors : np.array
        An array describing the model's anchor boxes by their width and height.

    Returns
    -------
    best_anchor : np.array
        The index of the best anchor box for each of the input boxes.
    """
    # Determine top-left and bottom-right corners of the bounding boxes in relation to the center
    boxes_width_height = np.expand_dims(boxes_width_height, -2)
    boxes_corners = get_min_max_corners(boxes_width_height)

    # Determine top-left and bottom-right of the anchor boxes, expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchors_corners = get_min_max_corners(anchors)

    # Compute the intersections between the anchor boxes and the bounding boxes
    intersect_top_left, intersect_bottom_right = intersect(boxes_corners, anchors_corners)
    # compute width and height, set to zero if negative (i.e. no intersection)
    intersect_wh = np.maximum(intersect_bottom_right - intersect_top_left, 0.)

    # Compute areas of the bounding boxes, the anchor boxes and their intersection
    box_area = compute_areas(boxes_width_height)
    anchor_area = compute_areas(anchors)
    intersect_area = compute_areas(intersect_wh)

    # Compute the intersection-over-union for each box
    iou = intersect_area / (box_area + anchor_area - intersect_area)

    # Find best anchor for each bounding box (anchor with maximum iou)
    best_anchor = np.argmax(iou, axis=-1)
    return best_anchor
