import cv2
from enum import Enum


class RGBColors(Enum):
    """Different RGB colors as tuples, used for default arguments throughout the code."""
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)


def draw_box_on_image(image, top_left, bottom_right, color=RGBColors.BLUE, thickness=None):
    """
    Draws a box defined by the top-left and bottom-right corners on the given image.

    Parameters
    ----------
    image : np.array
        The image to draw on.
    top_left : int-tuple
        The top-left corner of the box.
    bottom_right : int-tuple
        The bottom-right corner of the box.
    color : RGBColors or tuple of ints
        The color to use (default: blue).
    thickness : int
        The line-thickness. By default, the thickness is computed relative to the image size as (width + height) / 300.

    Returns
    -------
    np.array
        The image with the drawn on box.
    """
    if thickness is None:
        thickness = (image.shape[0] + image.shape[1]) // 300
    return cv2.rectangle(image, top_left, bottom_right, color.value, thickness)


def scale_text_to_width(text, desired_width, thickness=1, font=cv2.FONT_HERSHEY_DUPLEX,
                        min_scale=None, max_scale=None):
    """
    Compute the correct font scale to use on the given text with the specified font and thickness to achieve the
    desired width. Optionally, an upper and lower bound for the scale can be specified with min_scale and max_scale.

    Parameters
    ----------
    text : string
        The text to scale.
    desired_width : int
        The desired width.
    thickness : int
        The thickness used for the font.
    font : OpenCV font
        The font used.
    min_scale : double
        A lower bound for the font scale.
    max_scale : double
        An upper bound for the font scale.

    Returns
    -------
    scale : double
        The computed font scale.
    """
    text_width_scale_1 = cv2.getTextSize(text, fontFace=font, fontScale=1, thickness=thickness)[0][0]
    scale = desired_width / text_width_scale_1
    if min_scale is not None:
        scale = max(scale, min_scale)
    if max_scale is not None:
        scale = min(scale, max_scale)
    return scale


def draw_text_on_image(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, color=RGBColors.GREEN,
                       thickness=None, font_scale=None, max_width=None, image_shape=None):
    """
    Prints the specified text onto the given image at the given position.
    The label can be specified further through the font, the color, the thickness, and font_scale.
    By default, the thickness is compute as (width + height) / 600 if the image shape is given and 2 otherwise.
    If the font scale is not defined but a maximum width for the label is given, the font scale is computed using
    the scale_text_to_width function. If the maximum width is not defined either, the scale is set to 1.

    Parameters
    ----------
    image : np.array
        The image to draw on.
    text : string
        The text to draw.
    position : int-tuple
        The position to draw the text at (bottom-left corner).
    font : OpenCV font
        The font to use. (Default: Hershey-Simplex)
    color : RGBColors or tuple of ints
        The color to use. (Default: Green)
    thickness : int
        The thickness. If no thickness is specified, it is computed as (width + height) / 600 if the image shape is
        given and set to 2 otherwise.
    font_scale : double
        The font scale to use. If the font scale is not defined it is either computed from the max_width (if given)
        using the scale_text_to_width function or set to 1 otherwise.
    max_width : int
        The maximum width the text should require. Used to compute the font scale, if none is specified. For example,
        use the box width to annotate bounding boxes.
    image_shape : int-tuple
        The shape of the image to draw on. Used to compute the thickness, if none is specified.

    Returns
    -------
    np.array
        The image with the text drawn onto it.
    """
    if thickness is None:
        if image_shape is not None:
            thickness = max(2, (image_shape[0] + image_shape[1]) // 600)
        else:
            thickness = 2

    if font_scale is None:
        if max_width is None:
            font_scale = 1
        else:
            font_scale = scale_text_to_width(text, max_width, thickness, font)
    return cv2.putText(image, text, position, font, font_scale, color.value, thickness)
