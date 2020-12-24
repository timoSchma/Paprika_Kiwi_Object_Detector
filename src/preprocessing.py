import cv2
import numpy as np
import json
import pathlib
import copy

import src.annotated_image


class JSONUtil:
    """An utility class (used as namespace) for reading and writing json files."""

    @staticmethod
    def read(path):
        """
        Open the file at the given path, read the json content and return it.

        Parameters
        ----------
        path : path-like
            The path to the file to read from.

        Returns
        -------
        content : dict
            The content read from the specified file.
        """
        with open(path, 'r') as file:
            content = json.loads(file.read())
        return content

    @staticmethod
    def write(path, content):
        """
        Write the given content as json to the specified path.

        Parameters
        ----------
        path : path-like
            The path to write to.
        content : dict
            The content to write to the file.
        """
        # Create directories on output path if they don't exist
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as file:
            json.dump(content, file)

    @staticmethod
    def merge_jsons(path):
        """
        Merges all jsons at the specified path or a subdirectory into a single dict and returns this merged content.

        Parameters
        ----------
        path : path-like
            The path to an input folder whose json-files should be merged.

        Returns
        -------
        merged_contents : dict
            The combined content of all json-files at the given path.
        """
        # merge all json files in the input folder into a single dict
        contents = [JSONUtil.read(file) for file in pathlib.Path(path).rglob('*.json')]
        merged_contents = {key: content[key] for content in contents for key in content}
        return merged_contents


def fix_annotation_keys(annotation_dict):
    """
    A helper function to fix the keys in the annotation dict created by the VGG-Image-Annotator.
    Use only the file name of an image as its key.

    Parameters
    ----------
    annotation_dict : dict
        The dict containing the annotations.

    Returns
    -------
    dict
        The dict with fixed keys.
    """
    return {entry["filename"]: entry for _, entry in annotation_dict.items()}


def grayscale(img):
    """
    Function to plot the image in gray-scale.

    Parameters
    ----------
    img: image
        image to be displayed by the function

    Returns
    -------
    img: image
        image in grey-scale
    """

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def equalize_histogram(grey_img):
    """
    Function to equalize the intensity histogram in grey-scale.

    Parameters
    ----------
    grey_img: image
        image in grey-scale to be equalized by the function

    Returns
    -------
    cdf[grey_img]: matrix
        equalized intensity matrix (equalized image)
    """

    # Normalize the cumulative distribution function
    hist, bins = np.histogram(grey_img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    # Transform pixel intensities
    cdf_m = np.ma.masked_equal(cdf, 0)  # Ensure that we do not divide by zero (masking)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # Spread horizontal axis
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # Fill in with zero in case data is missing

    return cdf[grey_img]


def simple_thresholding(img, threshold):
    """
    Function to perform simple thresholding on the intensities of pixels

    Parameters
    ----------
    img: image
        image in grey-scale
    threshold: int
        threshold used to perform the binary classification of pixel intensities (0 or 255)

    Returns
    -------
    img: image
        image with converted intensities of pixels
    """
    return cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)


def adaptive_thresholding(img, size, c):
    """
    Function to perform adaptive thresholding on the intensities of pixels
    (should yield better results in case of variable lighting conditions in different areas of an image)

    Parameters
    ----------
    img: image
        image in grey-scale
    size: int
        size of the neighborhood that is used to calculate the threshold value
    c: int
        constant that is subtracted in the threshold calculation

    Returns
    -------
    img: image
        image with converted intensities of pixels
    """
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, size, c)


class Rotation:
    """This class is used as namespace for several rotation helper functions."""

    @staticmethod
    def rotation_matrix(rotation_center, angle):
        """
        Create a rotation matrix from a given rotation center and angle.
        Parameters
        ----------
        rotation_center: int-tuple
            The center to rotate around.
        angle: int
            The angle to rotate by.

        Returns
        -------
        rotation_matrix: np.array
            The computed rotation matrix.
        """
        return cv2.getRotationMatrix2D(center=tuple(rotation_center), angle=angle, scale=1.0)

    @staticmethod
    def rotate_image(image, angle, rotation_center):
        """
        Function to rotate the image by a certain angle.

        Parameters
        ----------
        image: image
            image to be rotated by the function
        angle: int
            angle by which the image will be rotated
        rotation_center: int-tuple
            rotation center

        Returns
        -------
        result: image
            rotated image
        """

        rotation_matrix = Rotation.rotation_matrix(rotation_center, angle)
        return cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    @staticmethod
    def rotate_point(point, angle, rotation_center):
        """
        Function to rotate a certain point by a certain angle.

        Parameters
        ----------
        point: int-tuple
            point to be rotated as (x, y)
        angle: int
            angle by which the point will be rotated
        rotation_center: int-tuple
            rotation center

        Returns
        -------
        result: numpy array
            rotated point
        """
        rotation_matrix = Rotation.rotation_matrix(rotation_center, angle)
        result = np.dot(rotation_matrix, np.append(point, 1))

        return result.astype(int)


def color_jittering(img, hue, sat, val):
    """
    Function to randomly adjust color & brightness of the image.

    Parameters
    ----------
    img : np.array
        Image to be adjusted.
    hue : double
        The range the hue can change.
    sat : double
        The range the saturation can change.
    val : double
        The range the value can change.

    Returns
    -------
    img : np.array of double
        Image with randomly adjusted hue, saturation, value in RGB format.
    """

    # convert RGB to HSV and get hue, saturation, value
    h, s, v = get_hue_sat_val(img)
    # randomly adjust hue, saturation, value
    h = hue_jittering(h, hue)
    s = sat_val_jittering(s, sat)
    v = sat_val_jittering(v, val)
    # convert adjusted HSV to RGB and return the image
    img = cv2.merge([h, s, v])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = img / 255
    return img


def get_hue_sat_val(img, is_interval_0_1=True):
    """
    Function that transforms an RGB input image into HSV and returns scaled hue, saturation, value of the image.

    Parameters
    ----------
    img : np.array
        RGB image
    is_interval_0_1 : boolean
        Whether the image is given with values in [0.0, 1.0] or [0, 255]

    Returns
    -------
    hue : np.array of double
        The scaled hue of the image
    saturation : np.array of double
        The scaled saturation of the image
    value : np.array of double
        The scaled value of the image
    """
    if is_interval_0_1:
        img = img * 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(img)
    # scale to the interval [0,1]
    hue = hue / 180.0  # hue is in the interval [0, 179]
    saturation = saturation / 255.0  # saturation & value in the interval [0, 255]
    value = value / 255.0
    return hue, saturation, value


def hue_jittering(hue, factor):
    """
    Function to randomly adjust the hue of an image.

    Parameters
    ----------
    hue : np.array of double
        Hue of an image
    factor : double
        Range in which the random factor to adjust hue can be in

    Returns
    -------
    hue : np.array of int
        Adjusted value for the hue
    """
    hue_factor = np.random.uniform(-factor, factor, 1)
    hue = hue + hue_factor
    hue = np.where(hue > 1, hue - 1, hue)
    hue = np.where(hue < 0, hue + 1, hue)
    hue = hue * 180
    hue = hue.astype('uint8')
    return hue


def sat_val_jittering(sat_val, factor):
    """
    Function to randomly adjust the saturation or value of an image

    Parameters
    ----------
    sat_val : np.array of double
        Saturation or value of an image
    factor : double
        Range in which the random factor to adjust saturation or value can be in

    Returns
    -------
    sat_val : np.array of int
        Adjusted value for the saturation or value
    """
    if np.random.random_sample() < 0.5:
        sat_val_factor = np.random.uniform(1, factor, 1)
    else:
        sat_val_factor = 1 / np.random.uniform(1, factor, 1)
    sat_val = sat_val * sat_val_factor
    sat_val = np.where(sat_val > 1, 1, sat_val)
    sat_val = np.where(sat_val < 0, 0, sat_val)
    sat_val = sat_val * 255
    sat_val = sat_val.astype('uint8')
    return sat_val


DEFAULT_THRESHOLDING_PARAMS = {
    "size": 7,  # size of neighborhood
    "c": 2  # constant
}


def preprocess_images(input_path, output_path, width, height, annotations, convert_to_grayscale=False,
                      should_equalize_histogram=False, thresholding=False, keep_aspect_ratio=False,
                      thresholding_params=DEFAULT_THRESHOLDING_PARAMS):
    """
    Preprocess all images at the given input path, stores the processed images at the output path.
    Preprocessing includes resizing to the specified size and optionally converting to grayscale, performing
    histogram-equalization and/or thresholding.

    Parameters
    ----------
    input_path: path-like
        The path to read the input images from.
    output_path: path-like
        The path to write the processed images to.
    width: int
        The desired width to resize the images to.
    height: int
        The desired height to resize the images to.
    annotations: dict
        A dict containing the annotations in VGG-Image-Annotator format.
    convert_to_grayscale: boolean
        Whether the image should be converted to grayscale
    should_equalize_histogram: boolean
        Whether histogram equalization should be performed.
    thresholding: boolean
        Whether thresholding should be performed.
    keep_aspect_ratio: boolean
        Set to True, to preserver the original aspect ratio when resizing and to False to stretch the image to the
        requested size.
    thresholding_params: dict
        An optional dict containing the thresholding parameters.

    Returns
    -------
    annotations: dict
        The updated annotations for the processed images (in VGG-Image-Annotator format).
    """
    # loop through all images in input folder and rescale size and BB
    for image_file in pathlib.Path(input_path).rglob('*.jpg'):
        image_name = image_file.name
        if image_name not in annotations:
            continue  # skip images without annotation

        # Read image and add annotations
        image = src.annotated_image.AnnotatedImage(image_path=image_file, annotation_dict=annotations)

        # Preprocess image:
        # Downscale
        if keep_aspect_ratio:
            image.resize_keep_aspect_ratio(width, height)
        else:
            image.resize(width, height)

        # Optional: convert to grayscale and perform histogram equalization and/or thresholding
        if convert_to_grayscale:
            image_data = image.image.image_data
            image_data = grayscale(image_data)

            if should_equalize_histogram:
                image_data = equalize_histogram(image_data)
            if thresholding:
                image_data = adaptive_thresholding(image_data, **thresholding_params)

            image.image.image_data = image_data

        # Update bounding boxes in annotations dict
        annotations[image_name]["regions"] = image.get_annotation_regions_as_json()

        # Write preprocessed image to output path
        image.save_image(pathlib.Path(output_path, image_name))

    return annotations


def rotate_all_images(input_path, output_path, annotations, angles):
    """
    For each images at the given input path, create new images rotated by all given angles and store them at the
    output path.

    Parameters
    ----------
    input_path: path-like
        The path to read the input images from.
    output_path: path-like
        The path to write the processed images to.
    annotations: dict
        A dict containing the annotations in VGG-Image-Annotator format.
    angles: list of int
        The angles to rotate by.

    Returns
    -------
    rotated_annotations: dict
        The annotations for all rotated images (in VGG-Image-Annotator format).
    """
    rotated_annotations = {}
    for image_file in pathlib.Path(input_path).rglob('*.jpg'):
        if image_file.name not in annotations:
            continue

        # Read image and annotations
        image = src.annotated_image.AnnotatedImage(image_path=image_file, annotation_dict=annotations)

        for angle in angles:
            image_name = f"{image_file.stem}_rot_{angle}.jpg"

            # copy and rotate image and save it to the output path
            rotated_image = copy.deepcopy(image)
            rotated_image.rotate(angle)
            rotated_image.save_image(pathlib.Path(output_path, image_name))

            # Generate new json annotations from the rotated boxes
            annotation_dict = copy.copy(annotations[image_file.name])
            annotation_dict["filename"] = image_name
            annotation_dict["regions"] = rotated_image.get_annotation_regions_as_json()
            rotated_annotations[image_name] = annotation_dict

    return rotated_annotations
