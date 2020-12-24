from keras import backend as K
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import control_flow_ops


def construct_grid(features):
    """
    Construct the grid based on output feature maps of network.

    Parameters
    ----------
    features: tensor
        shape=(..., 13, 13,...) or shape=(..., 26, 26,...)

    Returns
    -------
    grid: tensor
    grid_shape: tensor

    """
    grid_shape = K.shape(features)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(features))

    return grid, grid_shape


def yolo_head_base(features, anchors, num_classes, input_shape):
    """
    Reshape anchors and features and adjust predictions to each spatial grid point and anchor size.

    Parameters
    ----------
    features: tensor
    anchors: numpy.ndarray
        anchor boxes to be used
    num_classes: int
        number of classes to be predicted
    input_shape: tensor

    Returns
    -------
    grid: tensor
    features: tensor
    box_xy: tensor
        x- and y-position of bounding boxes
    box_wh: tensor
        width and height of bounding boxes
    """

    dtype = K.dtype(features)
    num_anchors = len(anchors)

    grid, grid_shape = construct_grid(features)

    # Reshape anchors and features
    anchors_shape = [1, 1, 1, num_anchors, 2]  # batch, height, width, num_anchors, box_params
    anchors_tensor = K.reshape(K.constant(anchors), anchors_shape)
    features_shape = [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5]
    features = K.reshape(features, features_shape)

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(features[..., :2]) + grid) / K.cast(grid_shape[::-1], dtype)
    box_wh = K.exp(features[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], dtype)

    return grid, features, box_xy, box_wh


def yolo_head_sigmoid(features, anchors, num_classes, input_shape):
    """
    Normalize box confidence and conditional class probability using sigmoid (logistic) function.

    Parameters
    ----------
    features: tensor
    anchors: numpy.ndarray
        anchor boxes to be used
    num_classes: int
        number of classes to be predicted
    input_shape: tensor

    Returns
    -------
    box_xy: tensor
        x- and y-position of bounding boxes
    box_wh: tensor
        width and height of bounding boxes
    box_confidence: tensor
        box confidence score --> objectness (probability that box contains an object)
    box_class_probabilities: tensor
        conditional class probabilities
    """

    _, features, box_xy, box_wh = yolo_head_base(features, anchors, num_classes, input_shape)

    box_confidence = K.sigmoid(features[..., 4:5])
    box_class_probabilities = K.sigmoid(features[..., 5:])

    return box_xy, box_wh, box_confidence, box_class_probabilities


def yolo_head(features, anchors, num_classes, input_shape, calc_loss=False):
    """
    Convert final layer features to bounding box parameters.

    Parameters
    ----------
    features: tensor
    anchors: numpy.ndarray
        anchor boxes to be used
    num_classes: int
        number of classes to be predicted
    input_shape: tensor
    calc_loss: boolean
        default value is False

    Returns
    -------
    tuple
        either the output if yolo_head_base (if calc_loss is True) or yolo_head_sigmoid
    """

    if calc_loss:
        return yolo_head_base(features, anchors, num_classes, input_shape)
    else:
        return yolo_head_sigmoid(features, anchors, num_classes, input_shape)


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """
    Get corrected boxes for input image (transform boxes back to original image).

    Parameters
    ----------
    box_xy: tensor
    box_wh: tensor
    input_shape: tensor
    image_shape: tensor

    Returns
    -------
    boxes: tensor
        scaled and transformed bounding boxes for original image
    """

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """
    Process conv layer output to return boxes and corresponding scores. For more details of boxes output see
    yolo_correct_boxes.

    Parameters
    ----------
    feats: tensor
    anchors: numpy.ndarray
        anchor boxes to be used
    num_classes: int
        number of classes to be predicted
    input_shape: tensor
    image_shape: tensor

    Returns
    -------
    boxes: tensor
        scaled and transformed bounding boxes for original image
    box_scores: tensor
        includes the box scores (box probability score)
    """
    box_xy, box_wh, box_confidence, box_class_probabilities = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probabilities
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs, anchors, num_classes, image_shape, max_boxes=20, score_threshold=.6, iou_threshold=.5):
    """
    Evaluate YOLO model on given input and return filtered boxes. This function is the main building block for
    inference.

    Parameters
    ----------
    yolo_outputs: tensor
    anchors: numpy.ndarray
    num_classes: int
    image_shape: tensor
    max_boxes: boolean
        default value is False
    score_threshold: float
        ignore all bounding boxes with confidence below this threshold
    iou_threshold: float
        threshold used for non-maximum suppression

    Returns
    -------
    boxes_: tensor
        post-processed bounding boxes
    scores_: tensor
        post-processed scores/confidence
    classes_: tensor
        post-processed class predictions
    """
    num_layers = 2  # len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for layer in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[layer],
                                                    anchors[anchor_mask[layer]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def box_iou(b1, b2):
    """
    Return iou tensor.

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4) with x, y, w, h
    b2: tensor, shape=(j, 4) with x, y, w, h

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    """
    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_layer_loss(y_true, anchors, yolo_outputs, grid_shapes, input_shape, num_classes, batch_size, ignore_thresh):
    """
    Calculates the YOLO loss of a layer based on classification, localization and confidence.
    Parameters
    ----------
    y_true: tensor
    anchors: numpy.ndarray
    yolo_outputs: tensor
    grid_shapes: tensor
    input_shape: tensor
    num_classes: integer
    batch_size: integer
    ignore_thresh: float

    Returns
    -------
    loss: float
    print_data: list
    """
    batch_size_float = K.cast(batch_size, K.dtype(yolo_outputs[0]))

    # Get object mask to filter out boxes with score 0
    object_mask = y_true[..., 4:5]  # the class scores (always 1 in training)
    object_mask_bool = K.cast(object_mask, 'bool')  # convert to boolean

    # Apply yolo head to get the prediction
    grid, raw_prediction, prediction_xy, prediction_wh = yolo_head(yolo_outputs, anchors, num_classes,
                                                                   input_shape, calc_loss=True)
    prediction_box = K.concatenate([prediction_xy, prediction_wh])

    # Darknet raw box to calculate loss.
    raw_true_xy = y_true[..., :2] * grid_shapes[::-1] - grid

    raw_true_wh = K.log(y_true[..., 2:4] / anchors * input_shape[::-1])
    raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf

    # = 2 - true box size
    box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]

    # Compute filter mask to ignore boxes whose iou is below the threshold
    filter_mask = filter_by_iou(y_true, object_mask_bool, prediction_box, ignore_thresh, batch_size)

    # Compute losses
    xy_loss, wh_loss = compute_box_loss(object_mask, raw_true_xy, raw_true_wh,
                                        raw_prediction, box_loss_scale, batch_size_float)
    confidence_loss = compute_confidence_loss(object_mask, filter_mask, raw_prediction, batch_size_float)
    class_loss = compute_class_loss(y_true, raw_prediction, object_mask, batch_size_float)

    loss = xy_loss + wh_loss + confidence_loss + class_loss

    print_data = [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(filter_mask)]

    return loss, print_data


def filter_by_iou(y_true, object_mask_bool, prediction_boxes, ignore_thresh, batch_size):
    """
    Filters boxes by IoU to ignore boxes, whose IoU is below the threshold

    Parameters
    ----------
    y_true: tensor
    object_mask_bool: tensor
    prediction_boxes: tensor
        predicted bounding boxes by output layer
    ignore_thresh: float
    batch_size: integer

    Returns
    -------
    filter_mask: tensor
    """
    dtype = K.dtype(y_true)
    filter_mask = tf.TensorArray(dtype, size=1, dynamic_size=True)

    def loop_body(image_id, mask):
        # get true boxes, filter by object score
        true_boxes = tf.boolean_mask(y_true[image_id, ..., 0:4], object_mask_bool[image_id, ..., 0])
        # compute iou between all predicted boxes and all true boxes
        iou_values = box_iou(prediction_boxes[image_id], true_boxes)
        # select maximum iou for each predicted box
        best_iou = K.max(iou_values, axis=-1)
        # filter out all predicted boxes whose maximum iou is below the threshold
        mask = mask.write(image_id, K.cast(best_iou < ignore_thresh, dtype))
        return image_id + 1, mask

    # loop over all images in the batch and set the filter mask to ignore boxes whose iou is below the threshold
    _, filter_mask = control_flow_ops.while_loop(cond=lambda image_id, *args: image_id < batch_size,
                                                 body=loop_body, loop_vars=[0, filter_mask])
    filter_mask = filter_mask.stack()
    filter_mask = K.expand_dims(filter_mask, -1)

    return filter_mask


def compute_box_loss(object_mask, true_xy, true_wh, prediction, box_loss_scale, batch_size):
    """
    Calculates the localization loss for each output layer.

    Parameters
    ----------
    object_mask: tensor
    true_xy: tensor
    true_wh: tensor
    prediction: tensor
    box_loss_scale
    batch_size: integer

    Returns
    -------
    xy_loss: float
    wh_loss: float
    """
    predicted_xy = prediction[..., 0:2]
    predicted_wh = prediction[..., 2:4]

    xy_cross_entropy = K.binary_crossentropy(target=true_xy, output=predicted_xy, from_logits=True)
    xy_loss = object_mask * box_loss_scale * xy_cross_entropy
    xy_loss = K.sum(xy_loss) / batch_size

    wh_loss = object_mask * box_loss_scale * 0.5 * K.square(true_wh - predicted_wh)
    wh_loss = K.sum(wh_loss) / batch_size

    return xy_loss, wh_loss


def compute_confidence_loss(object_mask, ignore_mask, raw_prediction, batch_size):
    """
    Calculates the confidence loss for each output layer.

    Parameters
    ----------
    object_mask: tensor
    ignore_mask: tensor
    raw_prediction: tensor
    batch_size: integer

    Returns
    -------
    confidence_loss: float
    """
    cross_entropy = K.binary_crossentropy(target=object_mask, output=raw_prediction[..., 4:5], from_logits=True)
    confidence_loss = object_mask * cross_entropy + (1 - object_mask) * cross_entropy * ignore_mask
    confidence_loss = K.sum(confidence_loss) / batch_size
    return confidence_loss


def compute_class_loss(y_true, raw_prediction, object_mask, batch_size):
    """
    Calculates the classification for each output layer.

    Parameters
    ----------
    y_true: tensor
    raw_prediction: tensor
    object_mask: tensor
    batch_size: integer

    Returns
    -------
    class_loss: float
    """
    true_class_probabilities = y_true[..., 5:]
    predicted_class_probabilities = raw_prediction[..., 5:]

    cross_entropy = K.binary_crossentropy(target=true_class_probabilities, output=predicted_class_probabilities,
                                          from_logits=True)
    class_loss = object_mask * cross_entropy
    class_loss = K.sum(class_loss) / batch_size
    return class_loss


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    """
    Return yolo_loss tensor

    Parameters
    ----------
    args : list of tensor
        The input tensor (output of the (tiny)-yolo body)
    anchors: array, shape=(N, 2), wh
    num_classes: integer
        the number of classes
    ignore_thresh: float
        the iou threshold whether to ignore object confidence loss
    print_loss : boolean
        whether to print the detailed loss

    Returns
    -------
    loss: tensor, shape=(1,)
        returns the total YOLO loss as sum of layer loss
    """
    num_layers = 2  # len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]

    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]

    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[layer])[1:3], K.dtype(y_true[0])) for layer in range(num_layers)]

    loss = 0

    batch_size = K.shape(yolo_outputs[0])[0]  # batch size, tensor

    for layer in range(num_layers):
        layer_loss, print_data = yolo_layer_loss(y_true=y_true[layer], anchors=anchors[anchor_mask[layer]],
                                                 yolo_outputs=yolo_outputs[layer], grid_shapes=grid_shapes[layer],
                                                 input_shape=input_shape, num_classes=num_classes,
                                                 batch_size=batch_size, ignore_thresh=ignore_thresh)
        loss += layer_loss

        if print_loss:
            loss = tf.Print(loss, print_data, message='loss: ')
    return loss
