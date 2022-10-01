import tensorflow as tf
import functools
import numpy as np


class Classifier(tf.keras.layers.Layer):
    def __init__(self, n_class=21, n_feature=2048, normalize=tf.keras.layers.BatchNormalization, activation=tf.keras.activations.relu, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.n_class = n_class
        self.n_feature = n_feature
        self.normalize = normalize
        self.activation = activation
        self.global_average_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.feature = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(self.n_feature // 2)
        self.logits = tf.keras.layers.Dense(self.n_class, activation=tf.keras.activations.softmax)
        self.norm1 = None
        if self.normalize is not None:
            self.norm1 = self.normalize()
        self.act1 = tf.keras.layers.Activation(self.activation, name="feature_act")

    def call(self, inputs):
        out = inputs
        out = self.global_average_pool(out)
        out = self.feature(out)
        out = self.dense(out)
        out = self.norm1(out)
        out = self.act1(out)
        logits = self.logits(out)
        return logits



