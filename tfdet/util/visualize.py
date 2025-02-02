import cv2
import numpy as np

def draw_bbox(x_true, bbox_true, y_true = None, mask_true = None, label = None, threshold = 0.5, mix_ratio = 0.5, method = cv2.INTER_LINEAR, probability = True, prefix = "", postfix = "", color = None, size_ratio = 1.):
    batch = True
    if np.ndim(x_true) not in [1, 4]:
        batch = False
        x_true = [x_true]
        bbox_true = [bbox_true]
        if y_true is not None:
            y_true = [y_true]
        if mask_true is not None:
            mask_true = [mask_true]

    result = []
    for batch_index in range(len(x_true)):
        image = np.array(x_true[batch_index])
        bbox = np.array(bbox_true[batch_index])
        h, w = np.shape(image)[:2]
        size = int(max(h, w) / 500 * size_ratio)
        font_size = max(h, w) / 1250 * size_ratio
        normalize_flag = np.max(image) <= 1
        y_color = (1, 1, 1) if normalize_flag else (255, 255, 255)
        
        valid_indices = np.where(0 < np.max(bbox, axis = -1))
        bbox = bbox[valid_indices]
        mask = np.array(mask_true[batch_index])[valid_indices] if mask_true is not None else None
        if y_true is not None:
            y = np.array(y_true[batch_index])[valid_indices]
            if np.shape(y)[-1] != 1:
                y_index = np.argmax(y, axis = -1)
                score = np.max(y, axis = -1)
            else:
                y_index = y[..., 0].astype(int)
                score = np.ones_like(y_index, dtype = np.float32)
        
        for index, rect in enumerate(bbox):
            bbox_color = color
            if color is None:
                bbox_color = np.random.random(size = 3) if normalize_flag else np.random.randint(0, 256, size = 3).astype(float)
            if np.max(rect) < 2:
                rect = np.round(np.multiply(rect, [w, h, w, h]))
            rect = tuple(rect.astype(int))
            
            if y_true is not None:
                name = label[y_index[index]] if label is not None else y_index[index]
                bbox_color = bbox_color[y_index[index]] if np.ndim(bbox_color) == 2 else bbox_color
                msg = "{0}{1}".format(prefix, name)
                if probability:
                    msg = "{0}:{1:.2f}".format(msg, score[index])
                msg = "{0}{1}".format(msg, postfix)
                text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_size, size)[0]
                font_pos = (rect[0], max(rect[1], text_size[1]))
            
            bbox_color = tuple(bbox_color) if np.ndim(bbox_color) == 1 else bbox_color
            cv2.rectangle(image, rect[:2], rect[-2:], bbox_color, size)
            if y_true is not None:
                cv2.rectangle(image, (font_pos[0], font_pos[1] - text_size[1]), (font_pos[0] + text_size[0], font_pos[1]), bbox_color, -1)
                cv2.putText(image, msg, font_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, y_color, size)

            if mask_true is not None:
                m = mask[index]
                mh, mw = np.shape(m)[:2]
                if mh == h and mw == w:
                    m = m[rect[1]:rect[3], rect[0]:rect[2]]
                m = cv2.resize(m, (min(rect[2], w) - rect[0], min(rect[3], h) - rect[1]), interpolation = method)
                m = np.where(np.greater(m, threshold), 1., 0.)
                m = np.tile(np.expand_dims(m, axis = -1), 3) * bbox_color
                crop = image[rect[1]:rect[3], rect[0]:rect[2]]
                image[rect[1]:rect[3], rect[0]:rect[2]] = np.where(np.greater(m, 0), crop * (1 - mix_ratio) + m * mix_ratio, crop)
        result.append(image)
    if not batch:# and len(x_true) == 1:
        result = result[0]
    else:
        result = np.array(result)
    return result
