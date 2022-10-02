import shutil
import os
import os.path as osp
import json
import cv2
import skimage.draw
import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import tfdet
import functools
import albumentations
import matplotlib.pyplot as plt


def parse(img_path_root, place_name_to_id, target_size=(512, 512), method=cv2.INTER_CUBIC):
    class_folder_list = os.listdir(img_path_root)
    total_image_list = []
    total_class_list = []

    for class_folder in tqdm(class_folder_list):
        img_file_list = os.listdir(osp.join(img_path_root, class_folder))
        class_id = place_name_to_id[class_folder]

        for img_file in img_file_list:
            img_full_path = osp.join(img_path_root, class_folder, img_file)
            image = cv2.cvtColor(cv2.imread(img_full_path, -1), cv2.COLOR_BGR2RGB)

            if target_size is not None:
                image = cv2.resize(image, target_size[::-1], interpolation=method)

            total_image_list.append(image)
            total_class_list.append(class_id)

    image_np = np.stack(total_image_list, axis=0)
    class_np = tf.one_hot(total_class_list, depth=(len(place_name_to_id) + 1)).numpy()

    return image_np, class_np


def parse_test(img_path_root, place_name_to_id, target_size=(512, 512)):
    class_folder_list = os.listdir(img_path_root)
    total_image_list = []
    total_class_list = []

    method = albumentations.Compose([
        albumentations.Normalize(),
        albumentations.Resize(target_size[0], target_size[1]),
    ])

    def aug_func(image):
        return method(image=image)["image"]

    for class_folder in tqdm(class_folder_list):
        img_file_list = os.listdir(osp.join(img_path_root, class_folder))
        class_id = place_name_to_id[class_folder]

        for img_file in img_file_list:
            img_full_path = osp.join(img_path_root, class_folder, img_file)
            image = cv2.cvtColor(cv2.imread(img_full_path, -1), cv2.COLOR_BGR2RGB)
            image = aug_func(image)
            total_image_list.append(image)
            total_class_list.append(class_id)

    image_np = np.stack(total_image_list, axis=0)
    class_np = tf.one_hot(total_class_list, depth=len(place_name_to_id)).numpy()

    return image_np, class_np


def parse_file_path(img_path_root, place_name_to_id, target_size=(512, 512), method=cv2.INTER_CUBIC):
    class_folder_list = os.listdir(img_path_root)
    total_image_list = []
    total_class_list = []

    for class_folder in tqdm(class_folder_list):
        img_file_list = os.listdir(osp.join(img_path_root, class_folder))
        class_id = place_name_to_id[class_folder]

        for img_file in img_file_list:
            img_full_path = osp.join(img_path_root, class_folder, img_file)
            total_image_list.append(img_full_path)
            total_class_list.append(class_id)

    class_np = tf.one_hot(total_class_list, depth=len(place_name_to_id)).numpy()

    return total_image_list, class_np


def load_image_and_label_from_path(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img, label


def pipe(data, func=None, batch_size=2, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((data['img'], data['label']))
    if callable(func):
        dataset = dataset.map(func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.batch(batch_size)

    # dataset = dataset.prefetch((batch_size * 2) + 1)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
