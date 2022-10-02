from lovewar_helper import place_data
from tfdet.model.classifier import place_classifier
import tensorflow as tf
import tfdet
import albumentations
from functools import partial
import numpy as np
from datetime import datetime
import os

'''
1. albumentations
https://github.com/albumentations-team/albumentations(https://albumentations.ai/docs/getting_started/image_augmentation/)

2. albumentations docs
https://albumentations.ai/docs/api_reference/augmentations/

3. recommend
albumentations.RandomBrightness / albumentations.RandomContrast / albumentations.ChannelShuffle /
albumentations.RandomCrop / albumentations.HorizontalFlip / albumentations.Rotate /
etc 
'''
def augment_train_data(train_ds, image_size, label_cnt, batch_size, shuffle=True):
    transforms = albumentations.Compose([
        albumentations.Normalize(),
        albumentations.Resize(image_size[0], image_size[1]),
        albumentations.RandomBrightness(0.2, p=0.5),
        albumentations.RandomContrast(0.2, p=0.5),
        albumentations.ChannelShuffle(p=0.5)
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

    ds_alb = train_ds.map(partial(process_data), num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
    ds_alb = ds_alb.map(set_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds_alb = ds_alb.repeat()
    if shuffle:
        ds_alb = ds_alb.shuffle(buffer_size=batch_size * 10)
    ds_alb = ds_alb.batch(batch_size)
    return ds_alb


def preprocess_train_data(train_ds, image_size, label_cnt, batch_size, shuffle=True):
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

    ds_alb = train_ds.map(partial(process_data), num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
    ds_alb = ds_alb.map(set_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds_alb = ds_alb.repeat()
    if shuffle:
        ds_alb = ds_alb.shuffle(buffer_size=batch_size * 10)
    ds_alb = ds_alb.batch(batch_size)
    return ds_alb


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
    use_augmentation = False
    learning_rate = 5e-5
    momentum = 0.9
    epochs = 100
    img_size = (512, 512, 3)
    img_wh = (512, 512)
    n_feature = 2048
    batch_size = 16

    output_name = 'p1_resnet50_pretrained_x'
    tr_img_root = "/media/ailab/c_hdd/machine_ws/dataset/love_war_place/data/train"
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
    tr_img, tr_class = place_data.parse_file_path(tr_img_root, class_dict)
    te_img, te_class = place_data.parse_file_path(te_img_root, class_dict)

    training_data = tf.data.Dataset.from_tensor_slices((tr_img, tr_class))
    validation_data = tf.data.Dataset.from_tensor_slices((te_img, te_class))

    training_data = training_data.map(place_data.load_image_and_label_from_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validation_data = validation_data.map(place_data.load_image_and_label_from_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if use_augmentation:
        training_data = augment_train_data(training_data, image_size=img_size, label_cnt=len(class_dict.keys()),
                                           batch_size=batch_size, shuffle=True)

    else:
        training_data = preprocess_train_data(training_data, image_size=img_size, label_cnt=len(class_dict.keys()),
                                              batch_size=batch_size, shuffle=True)

    validation_data = preprocess_val_data(validation_data, image_size=img_size, label_cnt=len(class_dict.keys()),
                                          batch_size=batch_size)

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

    os.makedirs("logs/{}/".format(output_name), exist_ok=True)
    os.makedirs("weight/{}/".format(output_name), exist_ok=True)

    logdir = "logs/{}/".format(output_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    checkpoint = tf.keras.callbacks.ModelCheckpoint("weight/{}/best_place_model.h5".format(output_name),
                                                    monitor='val_accuracy', verbose=1,
                                                    save_best_only=True, save_weights_only=True, mode='max', period=1)

    opt = tf.keras.optimizers.SGD(learning_rate, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(training_data, validation_data=validation_data, epochs=epochs, callbacks=[tensorboard_callback, checkpoint])
    model.save_weights("weight/{}/place_model.h5".format(output_name))

