import tensorflow as tf

from tfdet.core.assign import max_iou
from tfdet.core.loss import regularize as regularize_loss
from tfdet.core.util import map_fn
from .loss.yolo import score_accuracy, score_loss, logits_accuracy, logits_loss, regress_loss
from .target import yolo_target
from ..postprocess.yolo import FilterDetection

def train_model(input, score, logits, regress, anchors,
                assign = max_iou, sampling_count = None, positive_ratio = 0.5,
                proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, valid = False, performance_count = 5000,
                batch_size = 1, clip_ratio = 16 / 1000, 
                regularize = True, weight_decay = 1e-4, mode = "general", focal = True, alpha = .25, gamma = 1.5, class_weight = None, threshold = 0.5, missing_value = 0.):
    y_true = tf.keras.layers.Input(shape = (None, None), name = "y_true", dtype = score.dtype)
    bbox_true = tf.keras.layers.Input(shape = (None, 4), name = "bbox_true", dtype = regress.dtype)
    
    tile_anchors = tf.tile(tf.expand_dims(anchors, axis = 0), [tf.shape(input)[0], 1, 1])
    score_true, logits_true, _bbox_true, score_pred, logits_pred, bbox_pred = tf.keras.layers.Lambda(lambda args: map_fn(yolo_target, *args, dtype = (tf.int32, y_true.dtype, bbox_true.dtype, score.dtype, logits.dtype, regress.dtype), batch_size = batch_size, 
                                                                                                                         assign = assign, sampling_count = sampling_count, positive_ratio = positive_ratio, valid = valid, clip_ratio = clip_ratio), name = "yolo_target")([y_true, bbox_true, score, logits, regress, tile_anchors])
    
    _score_accuracy = tf.keras.layers.Lambda(lambda args: score_accuracy(*args, threshold = threshold, missing_value = missing_value), name = "score_accuracy")([score_true, score_pred])
    _logits_accuracy = tf.keras.layers.Lambda(lambda args: logits_accuracy(*args, missing_value = missing_value), name = "logits_accuracy")([score_true, logits_true, logits_pred])
    _score_loss = tf.keras.layers.Lambda(lambda args: score_loss(*args, focal = focal, missing_value = missing_value), name = "score_loss")([score_true, score_pred])
    _logits_loss = tf.keras.layers.Lambda(lambda args: logits_loss(*args, focal = focal, alpha = alpha, gamma = gamma, weight = class_weight, missing_value = missing_value), name = "logits_loss")([score_true, logits_true, logits_pred])
    _regress_loss = tf.keras.layers.Lambda(lambda args: regress_loss(*args, mode = mode, missing_value = missing_value), name = "regress_loss")([score_true, _bbox_true, bbox_pred])
    
    _score_accuracy = tf.expand_dims(_score_accuracy, axis = -1)
    _logits_accuracy = tf.expand_dims(_logits_accuracy, axis = -1)
    _score_loss = tf.expand_dims(_score_loss, axis = -1)
    _logits_loss = tf.expand_dims(_logits_loss, axis = -1)
    _regress_loss = tf.expand_dims(_regress_loss, axis = -1)
    
    
    y_pred, bbox_pred = FilterDetection(proposal_count = proposal_count, iou_threshold = iou_threshold, score_threshold = score_threshold, soft_nms = soft_nms, valid = valid, performance_count = performance_count,
                                        batch_size = batch_size, clip_ratio = clip_ratio)([score, logits, regress, anchors])
    model = tf.keras.Model([input, y_true, bbox_true], [y_pred, bbox_pred])
    
    model.add_metric(_score_accuracy, name = "score_accuracy", aggregation = "mean")
    model.add_metric(_logits_accuracy, name = "logits_accuracy", aggregation = "mean")
    model.add_metric(_score_loss, name = "score_loss", aggregation = "mean")
    model.add_metric(_logits_loss, name = "logits_loss", aggregation = "mean")
    model.add_metric(_regress_loss, name = "regress_loss", aggregation = "mean")
    model.add_loss(_score_loss)
    model.add_loss(_logits_loss)
    model.add_loss(_regress_loss)
    
    if regularize:
        model.add_loss(lambda: tf.reduce_sum(regularize_loss(model, weight_decay), keepdims = True, name = "regularize_loss"))
    return model