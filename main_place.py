from lovewar_helper import place_data
from tfdet.model.classifier import place_classifier
import tensorflow as tf
import tfdet
import albumentations
import numpy as np
from datetime import datetime
import os


if __name__ == '__main__':
    # backbone_list : resnet50, resnet101, mobilenet, vgg16
    backbone_type = "resnet50"
    use_pretrained_backbone = False
    use_augmentation = False
    learning_rate = 5e-5
    momentum = 0.9
    epochs = 100
    img_size = (512, 512, 3)
    n_feature = 2048

    output_name = 'p3_resnet101'
    tr_img_root = "/media/ailab/c_hdd/machine_ws/dataset/love_war_place/data/train"
    te_img_root = "/media/ailab/c_hdd/machine_ws/dataset/love_war_place/data/val"

    class_dict = {
        "car": 1,
        "front_of_buliding": 2,
        "hospital": 3,
        "house": 4,
        "indoor": 5,
        "restaurant": 6,
        "rooftop": 7,
        "street": 8
    }

    tr_img, tr_class = place_data.parse(tr_img_root, class_dict)
    te_img, te_class = place_data.parse(te_img_root, class_dict)
    tr_img = np.array(tr_img, dtype=np.float32) * 1 / 255
    te_img = np.array(te_img, dtype=np.float32) * 1 / 255

    x = tf.keras.layers.Input(shape=img_size)

    if use_pretrained_backbone:
        weights = "imagenet"

    else:
        weights = None

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

    out = place_classifier.Classifier(n_class=(len(class_dict) + 1), n_feature=n_feature)(feature)
    model = tf.keras.Model(inputs=x, outputs=out)
    model.summary()
    batch_size = 16

    if use_augmentation:
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
        method = [albumentations.RandomBrightness(0.2, p=0.5),
                  albumentations.RandomContrast(0.2, p=0.5),
                  albumentations.ChannelShuffle(p=0.5)]
        method = albumentations.Compose(method)

        def aug_func(image):
            return method(image=image)["image"]

        def func(img, label):
            shape = tf.shape(img)
            img = tf.reshape(
                tf.py_function(lambda *args: aug_func(args[0].numpy()), inp=[img], Tout=tf.float32),
                shape)
            return img, label
        tr_data = place_data.pipe({"img": tr_img, "label": tr_class}, func=func, batch_size=batch_size, shuffle=True)

    else:
        tr_data = place_data.pipe({"img": tr_img, "label": tr_class}, batch_size=batch_size, shuffle=True)

    te_data = place_data.pipe({"img": te_img, "label": te_class}, batch_size=batch_size, shuffle=False)

    os.makedirs("logs/{}/".format(output_name), exist_ok=True)
    os.makedirs("weight/{}/".format(output_name), exist_ok=True)

    logdir = "logs/{}/".format(output_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    checkpoint = tf.keras.callbacks.ModelCheckpoint("weight/{}/best_place_model.h5".format(output_name), monitor='val_accuracy', verbose=1,
                                                    save_best_only=True, save_weights_only=True, mode='max', period=1)

    opt = tf.keras.optimizers.SGD(learning_rate, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(tr_data, validation_data=te_data, epochs=epochs, callbacks=[tensorboard_callback, checkpoint])
    model.save_weights("weight/{}/place_model.h5".format(output_name))

