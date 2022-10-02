from lovewar_helper import place_data
from tfdet.model.classifier import place_classifier
import tensorflow as tf
import tfdet
import albumentations
from functools import partial
import numpy as np
from datetime import datetime
import os


def preprocess_val_data(val_ds, image_size, label_cnt, batch_size):
    transforms = albumentations.Compose([
        albumentations.Normalize(),
        albumentations.Resize(image_size[0], image_size[1]),
    ])

    def aug_fn(image):
        data = {"image": image}
        aug_data = transforms(**data)
        aug_img = aug_data["image"]
        aug_img = tf.cast(aug_img, tf.float32)
        return aug_img

    def process_data(image, label):
        aug_img = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)
        return aug_img, label

    def set_shapes(img, label, img_shape=image_size, label_count=label_cnt):
        img.set_shape(img_shape)
        label.set_shape(label_count)
        return img, label

    ds_alb = val_ds.map(partial(process_data), num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
    ds_alb = ds_alb.map(set_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)
    return ds_alb


if __name__ == '__main__':
    # backbone_list : resnet50, resnet101, mobilenet, vgg16
    backbone_type = "resnet50"
    use_pretrained_backbone = False
    learning_rate = 5e-5
    momentum = 0.9
    epochs = 100
    img_size = (512, 512, 3)
    img_wh = (512, 512)
    n_feature = 2048
    batch_size = 16

    output_name = 'p1_resnet50_pretrained_x'
    te_img_root = "/media/ailab/c_hdd/machine_ws/dataset/love_war_place/data/val"

    class_dict = {
        "car": 0,
        "front_of_buliding": 1,
        "hospital": 2,
        "house": 3,
        "indoor": 4,
        "restaurant": 5,
        "rooftop": 6,
        "street": 7
    }

    te_img, te_class = place_data.parse_test(te_img_root, class_dict)

    if use_pretrained_backbone:
        weights = "imagenet"

    else:
        weights = None

    x = tf.keras.layers.Input(shape=img_size)

    if backbone_type == "resnet50":
        feature = tfdet.model.backbone.resnet50(x, weights=weights)[-1]

    elif backbone_type == "resent101":
        feature = tfdet.model.backbone.resnet101(x, weights=weights)[-1]

    elif backbone_type == "vgg16":
        feature = tfdet.model.backbone.vgg16(x, weights=weights)[-1]

    elif backbone_type == "mobilenet":
        feature = tfdet.model.backbone.mobilenet(x, weights=weights)[-1]

    else:
        # default : resnet50
        feature = tfdet.model.backbone.resnet50(x, weights=weights)[-1]

    out = place_classifier.Classifier(n_class=len(class_dict), n_feature=n_feature)(feature)
    model = tf.keras.Model(inputs=x, outputs=out)
    model.summary()

    model.load_weights("weight/{}/best_place_model.h5".format(output_name))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    score = model.evaluate(te_img, te_class)

