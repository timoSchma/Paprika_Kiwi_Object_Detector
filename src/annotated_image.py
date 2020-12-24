from matplotlib import pyplot as plt
import cv2
import json
import pathlib
import numpy as np
import copy

import src.preprocessing
from src.visualization import RGBColors, draw_box_on_image, draw_text_on_image
from src.utils import Corner, Point


class BoundingBox:
    """
    A class to represent bounding boxes defined by their upper-left corner and size.
    Offers functionality like conversions between different formats to represent bounding boxes,
    reshaping the box (e.g. scaling, shifting, and rotating), and drawing the box.
    """

    def __init__(self, x=None, y=None, width=None, height=None):
        """
        Create a new bounding box defined by its upper-left corner and size.

        Parameters
        ----------
            x : int
                The x-coordinate of the upper-left corner.
            y : int
                The y-coordinate of the upper-left corner.
            width : int
                The box width.
            height : int
                The box height.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __str__(self):
        """
        Get the string representation of this bounding box.

        Returns
        -------
        string
            The string representation.
        """
        return f"x={self.x}, y={self.y}, width={self.width}, height={self.height}"

    def __copy__(self):
        """
        Copy this bounding box.

        Returns
        -------
        copy : BoundingBox
            The copy.
        """
        copied = type(self)()
        copied.__dict__.update(self.__dict__)
        return copied

    def __deepcopy__(self, memo=None):
        """
        Deep-copy this bounding box (is the same as copying since all variables are numbers)

        Parameters
        ----------
        memo : Dict
            unused

        Returns
        -------
        BoundingBox
            The copy.
        """
        return self.__copy__()

    def from_shape_attributes(self, shape_attributes):
        """
        Sets this bounding box from an annotation given as shape_attributes dict.

        Parameters
        ----------
        shape_attributes : Dict
            a dict with the keys 'x', 'y', 'width', and 'height'
        """
        self.x = shape_attributes["x"]
        self.y = shape_attributes["y"]
        self.width = shape_attributes["width"]
        self.height = shape_attributes["height"]

    def to_shape_attributes(self):
        """
        Converts this bounding box to an annotation in shape_attributes format, i.e. a dict with the keys
        'x', 'y', 'width', and 'height' (set by the corresponding member variables) and 'name': 'rect'.

        Returns
        -------
        shape_attributes : Dict
            This bounding box as shape_attributes dict.
        """
        return {"name": "rect",
                "x": int(self.x), "y": int(self.y),
                "width": int(self.width), "height": int(self.height)}

    def set_from_yolo_box(self, box, image_shape):
        """
        Sets this bounding box from a box output by yolo: a numpy array containing the top, left, bottom,
        and right coordinate

        Parameters
        ----------
        box : np.array
            Numpy array containing the top, left, bottom, and right coordinate.
        image_shape : tuple
            The shape of the image this box belongs to (used to fix out-of-bounds coordinates).
        """
        # Round to nearest integer
        top, left, bottom, right = np.rint(box).astype(int)

        # Make sure the box is within the image bounds
        left, top = max(0, left), max(0, top)
        right, bottom = min(right, image_shape[1]), min(bottom, image_shape[0])

        # Set from top-left and bottom-right corner
        self.set_from_corners([[left, top], [right, bottom]])

    def get_corner(self, corner):
        """
        Returns the requested corner of the bounding box.

        Parameters
        ----------
        corner : Corner
            The requested corner.

        Returns
        -------
        corner : Point
            The the requested corner as point.
        """
        if corner is Corner.TOP_LEFT:
            return Point(self.x, self.y)
        elif corner is Corner.TOP_RIGHT:
            return Point(self.x + self.width, self.y)
        elif corner is Corner.BOTTOM_LEFT:
            return Point(self.x, self.y + self.height)
        elif corner is Corner.BOTTOM_RIGHT:
            return Point(self.x + self.width, self.y + self.height)
        else:
            return None

    def set_from_corners(self, corners):
        """
        Sets this bounding box from an array-like of corners.

        Parameters
        ----------
        corners : np.array
            The corners of the box (should contain at least on upper & lower, and one left & right corner).
        """
        corners = np.array(corners)
        top_left = np.min(corners, axis=0)
        bottom_right = np.max(corners, axis=0)

        self.x, self.y = top_left
        self.width, self.height = bottom_right - top_left

    def get_center(self):
        """
        Gets the center of this bounding box.

        Returns
        -------
        center : Point
            The center as Point object
        """
        return Point(int(self.x + self.width * 0.5), int(self.y + self.height * 0.5))

    def scale(self, scaling_factors=None, old_shape=None, new_shape=None):
        """
        Scales this bounding box using either two scaling factors (one for each dimension)
        or the old and new shape of the image this box belongs to.

        Parameters
        ----------
        scaling_factors : tuple
            a tuple of the scaling factor for the height and the width. (Default value = None)
        old_shape : tuple
            the old shape of the underlying image. (Default value = None)
        new_shape : tuple
            the new shape of the underlying image. (Default value = None)
        """
        if scaling_factors is None:  # try computing the scaling_factors from old and new shape
            if old_shape is None or new_shape is None:
                return  # cannot compute scale if one of the shapes is None
            scaling_factors = [new_size / old_size for new_size, old_size in zip(new_shape, old_shape)]

        self.x = int(self.x * scaling_factors[1])
        self.y = int(self.y * scaling_factors[0])
        self.width = int(self.width * scaling_factors[1])
        self.height = int(self.height * scaling_factors[0])

    def shift(self, offset):
        """
        Shifts this bounding box according to the given offset.

        Parameters
        ----------
        offset : tuple
            A tuple of the offset in y and x direction.
        """
        self.x += offset[1]
        self.y += offset[0]

    def rotate(self, angle, rotation_center):
        """
        Rotates this bounding box with the given angle and rotation center.

        Parameters
        ----------
        angle : int
            The angle to rotate by.
        rotation_center : tuple
            The rotation center to use.
        """
        corners = np.array([self.get_corner(corner) for corner in Corner])
        rotated_corners = np.apply_along_axis(src.preprocessing.Rotation.rotate_point, axis=1, arr=corners,
                                              angle=angle, rotation_center=rotation_center)
        self.set_from_corners(rotated_corners)

    def draw_on_image(self, image, color=RGBColors.BLUE):
        """
        Draws this box on a given image using the specified color (blue by default).

        Parameters
        ----------
        image : np.array
            The image to draw on.
        color : RGBColors or 3-tuple
            The color (blue by default)

        Returns
        -------
        image : np.array
            The image with the drawn on bounding box.
        """
        top_left = self.get_corner(Corner.TOP_LEFT)
        bottom_right = self.get_corner(Corner.BOTTOM_RIGHT)
        return draw_box_on_image(image, top_left, bottom_right, color=color)

    def area(self):
        """
        Compute the box area.

        Returns
        -------
        area : number
            The area of this box.
        """
        return self.width * self.height

    def intersect(self, other):
        """
        Compute the intersection between this box and another box.

        Parameters
        ----------
        other : BoundingBox
            The other box to intersect with.

        Returns
        -------
        intersection : BoundingBox
            The intersection as bounding box.
        """
        top_left = np.maximum([self.x, self.y], [other.x, other.y])
        bottom_right = np.minimum([self.x + self.width, self.y + self.height],
                                  [other.x + other.width, other.y + other.height])
        x, y = top_left
        width, height = bottom_right - top_left
        intersection = BoundingBox(x, y, width, height)
        return intersection

    def iou(self, other):
        """
        Compute the intersection over union for this box with another box.

        Parameters
        ----------
        other : BoundingBox
            The other box.

        Returns
        -------
        iou : double
            The iou.
        """
        intersection = self.intersect(other).area()
        union = self.area() + other.area() - intersection
        if union == 0:
            return 1
        return intersection / union


class Annotation:
    """
    A class to represent image annotations for object detection consisting of a bounding box and
    the corresponding label and score.
    Offers functionality like conversions between different formats to represent annotations,
    reshaping the annotated bounding box (e.g. scaling, shifting, and rotating), and drawing the annotation.
    """

    def __init__(self, x=None, y=None, width=None, height=None, label=None, score=None, region_dict=None):
        """
        Create a new annotation object.
        The parameters can either be passed explicitly or as a region_dict defining this annotation.

        Parameters
        ----------
        x : int
            The x-coordinate of the upper-left corner of the bounding box.
        y : int
            The y-coordinate of the upper-left corner of the bounding box.
        width : int
            The box width.
        height : int
            The box height.
        label : int
            The class label.
        score : double
            The score of this annotation. (1.0 during training, [0.0, 1.0] for predictions)
        region_dict : Dict
            A dict describing this annotation, should contain a 'shape_attributes' dict describing the bounding box
            and a 'region_attributes' dict containing the class label.
        """
        self.bounding_box = BoundingBox(x, y, width, height)
        self.label = label
        self.score = score

        if region_dict is not None:
            self.from_region_dict(region_dict)

    def __copy__(self):
        """
        Copy this annotation.

        Returns
        -------
        copy: Annotation
            The copy.
        """
        copied = type(self)()
        copied.__dict__.update(self.__dict__)
        return copied

    def __deepcopy__(self, memo=None):
        """
        Deep-copy this annotation.

        Parameters
        ----------
        memo : Dict
            unused

        Returns
        -------
        copy: Annotation
            The copy.
        """
        copied = self.__copy__()
        copied.bounding_box = copy.deepcopy(self.bounding_box)
        return copied

    def __str__(self):
        """
        Get the string representation of this annotation.

        Returns
        -------
        string:
            The string representation
        """
        return f"label={self.label}, score={self.score}, bounding_box={self.bounding_box}"

    def from_region_dict(self, region_dict):
        """
        Sets this bounding box from an annotation given as region_dict.

        Parameters
        ----------
        region_dict : Dict
            The region dict describing a single annotated box in VGG-Image-Annotator format.
        """
        self.bounding_box.from_shape_attributes(region_dict["shape_attributes"])
        try:
            self.label = region_dict["region_attributes"]["Objects"]
        except KeyError as e:
            raise KeyError(f"Invalid key {e} for region_attributes with the following keys: "
                           f"{list(region_dict['region_attributes'].keys())}")
        self.score = 1

    def to_region_dict(self):
        """
        Converts this annotation to the region_dict format.

        Returns
        -------
        shape_attributes : Dict
            This annotation in region_dict format.
        """
        shape_attributes = self.bounding_box.to_shape_attributes()
        return {"shape_attributes": shape_attributes, "region_attributes": {"Objects": self.label}}

    def set_from_yolo_prediction(self, box, image_shape, score, class_id, class_names):
        """
        Sets this annotation from a yolo-prediction consisting of the box in yolo-format, the score and the id of the
        predicted class.

        Parameters
        ----------
        box : np.array
            Numpy array containing the top, left, bottom, and right coordinate.
        image_shape : tuple
            The shape of the image this annotation belongs to.
        score : double
            The score of this annotation. (1.0 during training, [0.0, 1.0] for predictions)
        class_id : int
            The class id for this annotation.
        class_names : List[string]
            A list of class-names (used to get the class label from the class id).
        """
        self.bounding_box.set_from_yolo_box(box, image_shape)
        self.score = score
        self.label = class_names[class_id]

    def get_in_yolo_format(self, class_names):
        """
        Converts this annotation to yolo format, i.e. an array with the following five entries:
        center-x, center-y, width, height, class-id

        Parameters
        ----------
        class_names : List[String]
            a list of class names to retrieve the class id from

        Returns
        -------
        yolo_annotation : np.array
            An array with five entry corresponding to the center-x, center-y, width, height, class-id of the box.

        """
        center_x, center_y = self.get_center()
        class_id = class_names.index(self.label)

        return np.array([center_x, center_y, self.bounding_box.width, self.bounding_box.height, class_id])

    def get_corner(self, corner):
        """
        Returns the requested corner of the bounding box.

        Parameters
        ----------
        corner : Corner
            The requested corner.

        Returns
        -------
        corner : Point
            The the requested corner as point.
        """
        return self.bounding_box.get_corner(corner)

    def set_from_corners(self, corners):
        """
        Sets the bounding box from an array-like of corners.

        Parameters
        ----------
        corners : np.array
            The corners of the box (should contain at least on upper & lower, and one left & right corner).
        """
        self.bounding_box.set_from_corners(corners)

    def get_center(self):
        """
        Gets the center of the bounding box.

        Returns
        -------
        center : Point
            The center as Point object
        """
        return self.bounding_box.get_center()

    def scale(self, scaling_factors=None, old_shape=None, new_shape=None):
        """
        Scales the bounding box using either two scaling factors (one for each dimension)
        or the old and new shape of the image this box belongs to.

        Parameters
        ----------
        scaling_factors : tuple
            a tuple of the scaling factor for the height and the width. (Default value = None)
        old_shape : tuple
            the old shape of the underlying image. (Default value = None)
        new_shape : tuple
            the new shape of the underlying image. (Default value = None)
        """
        self.bounding_box.scale(scaling_factors, old_shape, new_shape)

    def shift(self, offset):
        """
        Shifts the bounding box according to the given offset.

        Parameters
        ----------
        offset : tuple
            A tuple of the offset in y and x direction.
        """
        self.bounding_box.shift(offset)

    def rotate(self, angle, rotation_center):
        """
        Rotates the bounding box with the given angle and rotation center.

        Parameters
        ----------
        angle : int
            The angle to rotate by.
        rotation_center : tuple
            The rotation center to use.
        """
        self.bounding_box.rotate(angle, rotation_center)

    def draw_on_image(self, image, class_colors=None):
        """
        Draws this annotation (box, label and score) on a given image.
        If the class colors are given, the color corresponding to the label is used.

        Parameters
        ----------
        image : np.array
            The image to draw on.
        class_colors : Dict
            A dict mapping class labels to colors. (Default value = None)

        Returns
        -------
        image : np.array
            The image with the drawn on annotation.
        """
        if class_colors is not None and self.label in class_colors:
            color = class_colors[self.label]
        else:
            color = RGBColors.BLUE

        image = self.bounding_box.draw_on_image(image, color)

        text = f"{self.label} [{self.score:.2f}]"
        text_position = (self.bounding_box.x, self.bounding_box.y - 10)
        image = draw_text_on_image(image, text, text_position, color=color, max_width=self.bounding_box.width,
                                   image_shape=image.shape)
        return image

    def area(self):
        """
        Compute the box area.

        Returns
        -------
        area : number
            The area of this box.
        """
        return self.bounding_box.area()

    def intersect(self, other):
        """
        Compute the intersection between this box and another box.

        Parameters
        ----------
        other : BoundingBox or Annotation
            The other box to intersect with.

        Returns
        -------
        intersection : BoundingBox
            The intersection as bounding box.
        """
        if isinstance(other, Annotation):
            other = other.bounding_box
        return self.bounding_box.intersect(other)

    def iou(self, other):
        """
        Compute the intersection over union for this box with another box.

        Parameters
        ----------
        other : BoundingBox or Annotation
            The other box.

        Returns
        -------
        iou : double
            The iou.
        """
        if isinstance(other, Annotation):
            other = other.bounding_box
        return self.bounding_box.iou(other)


class Image:
    """
    This class is a wrapper to an image represented as numpy array.
    It offers additional functionality like loading/saving from/to files and reshaping (e.g. scaling,
    shifting, and rotating).
    """

    def __init__(self, rgb_image=None, path=None):
        """
        Creates a new image object. If the given path is not None, the contained image-data is loaded
        from this corresponding file.

        Parameters
        ----------
        rgb_image : np.array
            An RGB-image.
        path : path-like
            A path to an image file.
        """
        self.image_data = rgb_image

        if path is not None:
            self.load_image(path)

    def __copy__(self):
        """
        Copy this image.

        Returns
        -------
        copy : Image
            The copy.
        """
        copied = type(self)()
        copied.__dict__.update(self.__dict__)
        return copied

    def __deepcopy__(self, memo=None):
        """
        Deep-copy this image including the underlying image-data array.

        Parameters
        ----------
        memo : Dict
            unused

        Returns
        -------
        copy : Image
            The copy.
        """
        copied = self.__copy__()
        copied.image_data = self.image_data.copy()
        return copied

    def load_image(self, path):
        """
        Loads the contained image from the given path using open-cv. Also performs the conversion from BGR to RGB.
        Might throw a FileNotFoundError if the given path is incorrect or a ValueError if the image cannot be
        loaded correctly.

        Parameters
        ----------
        path : path-like
            The path to load the image from.
        """
        if not pathlib.Path(path).is_file():
            raise FileNotFoundError(f"Error: no such file '{path}'.")
        try:
            bgr_image = cv2.imread(str(path))
            self.image_data = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f"Error reading image at '{path}': {e}")

    def save_image(self, path):
        """
        Saves this image to the given path. If necessary, missing parents on the path are created.

        Parameters
        ----------
        path : path-like
            The path to save this image to.
        """
        bgr_image = cv2.cvtColor(self.image_data, cv2.COLOR_RGB2BGR)
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), bgr_image)

    def show(self):
        """Displays this image."""
        if len(self.image_data.shape) == 2 or self.image_data.shape[2] == 1:
            vmax = 1 if isinstance(self.image_data, np.floating) else 255
            plt.imshow(self.image_data, cmap='gray', vmin=0, vmax=vmax)
        else:
            plt.imshow(self.image_data)
        plt.show()

    def resize(self, width, height):
        """
        Resizes this image using open-cv and inter-cubic interpolation to the given size.
        Does not consider the original aspect-ratio. The image will be stretched as necessary.

        Parameters
        ----------
        width : int
            The width to resize to.
        height : int
            The height to resize to.
        """
        self.image_data = cv2.resize(self.image_data, (width, height), interpolation=cv2.INTER_CUBIC)

    def __scale_shape(self, scale):
        """
        Private method to convert a given scale to the new image shape.

        Parameters
        ----------
        scale : double
            The scale to scale the image with.

        Returns
        -------
        scaled_shape : int 2-tuple
            A 2-tuple containing the scale image shape.
        """
        shape = self.image_data.shape[:2]
        return tuple(int(dim * scale) for dim in shape)

    def scale(self, scale):
        """
        Scales this image with the given scale.

        Parameters
        ----------
        scale : double
            The scale to use.
        """
        height, width = self.__scale_shape(scale)
        self.resize(width, height)

    def compute_parameters_for_resize_with_ratio(self, width, height):
        """
        Helper function to compute the following three parameters: scale, offset, and border
        for resizing this image while retaining the aspect ratio.

        Parameters
        ----------
        width : int
            The width to resize to.
        height : int
            The height to resize to.

        Returns
        -------
        scale : double
            The scale to scale the image with.
        offset : 2-tuple of int
            The offset to place the scaled image centered within the padded area.
        border : 4-tuple of int
            The size of the padded border in top, bottom, left, and right direction.
        """
        desired_shape = np.array([height, width])
        old_shape = np.array(self.image_data.shape[:2])

        scale = np.min(desired_shape / old_shape)
        scaled_shape = self.__scale_shape(scale)

        delta_shape = desired_shape - scaled_shape
        offset = delta_shape // 2

        border_top, border_left = offset[:]
        border_bottom, border_right = (delta_shape - offset)[:]

        border = border_top, border_bottom, border_left, border_right

        return scale, offset, border

    def resize_keep_aspect_ratio(self, width, height, padding_color=RGBColors.BLACK):
        """
        Resize this image while retaining the aspect ratio. Instead of stretching the image, it is padded to the
        requested size using the specified padding color.

        Parameters
        ----------
        width : int
            The width to resize to.
        height : int
            The height to resize to.
        padding_color : RGBColors or int 3-tuple
            The color to pad with (default is black)
        """
        scale, _, border = self.compute_parameters_for_resize_with_ratio(width, height)

        self.scale(scale)
        self.image_data = cv2.copyMakeBorder(self.image_data, *border, cv2.BORDER_CONSTANT, value=padding_color.value)

    def rotate(self, angle):
        """
        Rotated this image by the specified angle around its center.

        Parameters
        ----------
        angle : int
            The angle to rotate by.
        """
        rotation_center = np.array(self.image_data.shape[1:: -1]) / 2
        self.image_data = src.preprocessing.Rotation.rotate_image(self.image_data, angle, rotation_center)


class AnnotatedImage:
    """
    This class represents an annotated image combining an image object with a list of annotations for object detection.
    """

    def __init__(self, rgb_image=None, annotations=None, image_path=None, annotation_path=None, annotation_dict=None):
        """
        Create a new annotated image.

        Parameters
        ----------
        rgb_image : np.array
            The underlying image data as numpy-array. Expected in RGB-order.
        annotations : List[Annotation]
            A list of annotations for this image.
        image_path : path-like
            A path to load the image from.
        annotation_path : path-like
            A path to load the annotations from (only used if image_path is given).
        annotation_dict : Dict
            A dict to extract the annotations from (only used if image_path is given).
        """
        self.image = Image(rgb_image, image_path)
        self.annotations = [] if annotations is None else annotations

        if image_path is not None:
            image_name = pathlib.Path(image_path).name
            if annotation_dict is not None:
                self.annotations_from_dict(annotation_dict, image_name)
            elif annotation_path is not None:
                self.read_annotations_from_json(annotation_path, image_name)

    def __copy__(self):
        """
        Copy this annotated image.

        Returns
        -------
        copy : AnnotatedImage
            The copy.
        """
        copied = type(self)()
        copied.__dict__.update(self.__dict__)
        return copied

    def __deepcopy__(self, memo=None):
        """
        Deep-copy this annotated image including the underlying image and list of annotations.

        Parameters
        ----------
        memo : Dict
            unused

        Returns
        -------
        copy : AnnotatedImage
            The copy.
        """
        copied = self.__copy__()
        copied.image = copy.deepcopy(self.image)
        copied.annotations = copy.deepcopy(self.annotations)
        return copied

    def load_image(self, path):
        """
        Load the underlying image from the given path.

        Parameters
        ----------
        path : path-like
            The path to load the image from.
        """
        self.image.load_image(path)

    def save_image(self, path, with_annotations=True, class_colors=None):
        """
        Save this annotated image to the given path.
        The image can be saved either with (default) or without the annotations. When saving with annotation,
        the colors to annotate with can be specified by class_colors.

        Parameters
        ----------
        path : path-like
            The path to save to.
        with_annotations : boolean
            Whether to save with or without annotations. (Default value = True)
        class_colors : Dict
            A dict mapping class labels to colors, used to color the annotations. (Default value = None)
        """
        if with_annotations:
            annotated_image = self.apply_annotations_to_image(class_colors)
            annotated_image.save_image(path)
        self.image.save_image(path)

    def read_annotations_from_json(self, path, image_name):
        """
        Reads the annotations for this annotated image from a json-file at the given path using the given image name
        as key. The json-file is expected to be in the VGG-Image-Annotator format.

        Parameters
        ----------
        path : path-like
            The path to the json-file containing the annotations.
        image_name : string
            The image name to use as key in the annotation file. (Usually, this is the file name of the
            corresponding image including the file-extension.)
        """
        with open(path, 'r') as json_file:
            json_dict = json.loads(json_file.read())
            self.annotations_from_dict(json_dict, image_name)

    def annotations_from_dict(self, json_dict, image_name):
        """
        Helper function to load and set the annotations for this annotated image from a given dict in the
        VGG-Image-Annotator format.

        Parameters
        ----------
        json_dict : Dict
            The annotations loaded from a json created by VGG-Image-Annotator.
        image_name : string
            The image name to use as key. (Usually, this is the file name of the corresponding image
            including the file-extension.)
        """
        annotations_dict = json_dict[image_name]
        self.annotations = [Annotation(region_dict=region_dict)
                            for region_dict in annotations_dict["regions"]]

    @staticmethod
    def from_yolo_prediction(prediction, input_image, class_names):
        """
        Create a new annotated image from a yolo-prediction, the corresponding input image and a list of class names.

        Parameters
        ----------
        prediction : List of 3-tuples (box, score, class_id) : (np.array, double, int)
            The prediction made by the yolo-model.
        input_image : np.array
            The corresponding image the prediction was made on.
        class_names : List[string]
            A list of class labels (used to convert the predicted class id to a class name).

        Returns
        -------
        A new annotated image with the given input image and the annotations extracted from the yolo prediction.
        """
        annotated_image = AnnotatedImage(rgb_image=input_image.copy(), annotations=[])

        for box, score, class_id in zip(*prediction):
            annotation = Annotation()
            annotation.set_from_yolo_prediction(box, input_image.shape, score, class_id, class_names)
            annotated_image.annotations.append(annotation)

        return annotated_image

    def get_annotation_regions_as_json(self):
        """
        Converts the annotations to a list of region dicts in the same format as output by the VGG Image Annotator.

        Returns
        -------
        regions_list: List[Dict]
            The list of region dicts.
        """
        regions_list = [annotation.to_region_dict() for annotation in self.annotations]
        return regions_list

    def get_annotations_in_yolo_format(self, class_names, shuffle=True, max_boxes=20):
        """
        Converts this annotated image to the format expected by the yolo model, i.e.
        - each annotation is represented as 5-tuple x-min, y-min, x-max, y-max, class-id
        - all tuples are stacked in a single numpy array of shape max_boxes x 5
        - at most max_boxes annotations are returned, unused lines are filled with zeros
        - use the shuffle parameter, to decide whether the annotations are shuffled (especially relevant if
          #annotations > max_boxes, shuffling happens before the cutoff).

        Parameters
        ----------
        class_names : List[string]
            A list containing the class names (used to convert the class label to id).
        shuffle : boolean
            Whether the annotations should be shuffled. (Default value = True)
        max_boxes : int
            The maximum number of boxes to return. (Default value = 20)

        Returns
        -------
        yolo_annotations : np.array
            The annotations in yolo-format, each line describes one annotation.
        """
        # convert each annotation to yolo format (i.e. a 5-tuple of center-x, center-y, width, height, class-id)
        yolo_annotations = np.array([annotation.get_in_yolo_format(class_names)
                                     for annotation in self.annotations])

        # optional: shuffle the annotations (shuffles only along the first axis)
        if shuffle:
            np.random.shuffle(yolo_annotations)

        # reshape to (max_boxes, 5), if number of annotations > max_boxes: discard annotations above this limit
        # if number of annotations < max_boxes: add zero-annotations to fill up the array
        box_count = yolo_annotations.shape[0]
        if box_count >= max_boxes:
            yolo_annotations = yolo_annotations[:max_boxes]
        else:
            zero_annotations = np.zeros((max_boxes, 5))
            zero_annotations[:box_count] = yolo_annotations
            yolo_annotations = zero_annotations

        return yolo_annotations

    def apply_annotations_to_image(self, class_colors=None):
        """
        Draws the annotation on a copy of the underlying image.

        Parameters
        ----------
        class_colors : Dict
            A dict specifying the colors to use for each class. (Default value = None)

        Returns
        -------
        annotated_image : Image
            A copy of the underlying image with the annotations drawn onto it.

        """
        annotated_image = copy.deepcopy(self.image)
        image_data = annotated_image.image_data
        for annotation in self.annotations:
            image_data = annotation.draw_on_image(image_data, class_colors)
        annotated_image.image_data = image_data
        return annotated_image

    def show(self, class_colors=None, with_annotations=True):
        """
        Displays this annotated image with (default) or without annotations.
        Optionally, the colors used for each class can be specified.

        Parameters
        ----------
        class_colors : Dict
            A dict specifying the colors to use for each class. (Default value = None)
        with_annotations : boolean
            Whether to show the image with or without annotations. (Default value = True)
        """
        if with_annotations:
            annotated_image = self.apply_annotations_to_image(class_colors)
            annotated_image.show()
        else:
            self.image.show()

    def rescale_annotations(self, scaling_factor=None, old_shape=None, new_shape=None):
        """
        Helper function to rescale all annotations.

        Parameters
        ----------
        scaling_factor : 2-tuple of double
            The scaling factor for the y and x direction.
        new_shape : int-tuple
            The new shape of the underlying image.
        old_shape : int-tuple
            The old shape of the underlying image.
        """
        for annotation in self.annotations:
            annotation.scale(scaling_factor, old_shape, new_shape)

    def shift_annotations(self, offset):
        """
        Shifts all annotations by the given offset.

        Parameters
        ----------
        offset : 2-tuple of ints
            A 2-tuple defining the y and x offset (in pixels)
        """
        for annotation in self.annotations:
            annotation.shift(offset)

    def scale(self, scale):
        """
        Scales this annotated image by the given scale

        Parameters
        ----------
        scale : 2-tuple of doubles
            A scalar value to scale by.
        """
        self.image.scale(scale)
        self.rescale_annotations(scaling_factor=(scale, scale))

    def resize(self, width=None, height=None):
        """
        Resizes this annotated image to the given size, stretching it as necessary. The annotations are resized
        together with the underlying image.

        Parameters
        ----------
        width : int
            The width to resize to. (Default value = None)
        height : int
            The height to resize to. (Default value = None)
        """
        if width is None and height is None:
            return self.image
        elif width is None or height is None:
            original_height, original_width = self.image.image_data.shape[:2]
            if width is None:
                scale = height / original_height
            else:
                scale = width / original_width
            self.scale(scale)

        self.rescale_annotations(old_shape=self.image.image_data.shape[:2], new_shape=(height, width))
        self.image.resize(width, height)

    def resize_keep_aspect_ratio(self, width, height, padding_color=RGBColors.BLACK):
        """
        Resizes this annotated image to the given size. However, instead of stretching the image, the original aspect
        ratio is preserved and the resulting image padded with the given padding color (black by default).

        Parameters
        ----------
        width : int
            The width to resize to.
        height : int
            The height to resize to.
        padding_color : RGBColors or tuple of ints
            The color to use for padding (default: black).
        """
        scale, offset, _ = self.image.compute_parameters_for_resize_with_ratio(width, height)
        self.image.resize_keep_aspect_ratio(width, height, padding_color)

        self.rescale_annotations(scaling_factor=(scale, scale))
        self.shift_annotations(offset)

    def rotate(self, angle):
        """
        Rotate this annotated image by the given angle around its center.

        Parameters
        ----------
        angle : int
            The angle to rotate by.
        """
        rotation_center = np.array(self.image.image_data.shape[1:: -1]) / 2
        self.image.rotate(angle)
        for annotation in self.annotations:
            annotation.rotate(angle, rotation_center)
