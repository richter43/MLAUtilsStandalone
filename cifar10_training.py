#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import numpy as np
import tensorflow as tf
from scipy import io as spio
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import sys
sys.path.append("src/")
from resnet import ResNet

ROOTDIR_DATA = "/content/drive/MyDrive/Teaching&Thesis/Teaching_dataset/"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'


# # Load Corrupted Cifar10

# In[3]:


def process_data(image, label):
#     image = tf.image.resize(image, (32, 32))
    return tf.cast(image, tf.float32)/255., tf.one_hot(label, 10, name='label', axis=-1)

# shot_noise = np.load(os.path.join(ROOTDIR_DATA, "CIFAR-10-C/shot_noise.npy"))
# shot_noise_5 = shot_noise[40000:]

(cifar10_train, cifar10_test), cifar10_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# mean_image = np.mean(x_train, axis=0)
# num_classes = 10
# x_train = x_train - mean_image / 255
# x_test = x_test - mean_image / 255
# x_train = x_train.astype("float16")
# x_train = x_train.astype("float16")
# y_train = tf.keras.utils.to_categorical(y_train, num_classes)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes)
cifar10_train = cifar10_train.map(
    process_data, num_parallel_calls=tf.data.AUTOTUNE)
cifar10_train = cifar10_train.cache()
cifar10_train = cifar10_train.shuffle(cifar10_info.splits['train'].num_examples)
cifar10_train = cifar10_train.batch(128)
cifar10_train = cifar10_train.prefetch(tf.data.AUTOTUNE)

cifar10_test = cifar10_test.map(
    process_data, num_parallel_calls=tf.data.AUTOTUNE)
cifar10_test = cifar10_test.cache()
cifar10_test = cifar10_test.batch(128)
cifar10_test = cifar10_test.prefetch(tf.data.AUTOTUNE)


# In[ ]:


# def show_images_from_npy(array, cmap=None):
#     fig, ax = plt.subplots(1, 5)
#     ax = ax.ravel()
#     j = 0
#     for idx in np.random.randint(0, array.shape[0], 5):
#         ax[j].imshow(array[idx], cmap=cmap)
#         ax[j].axis('off')
#         j += 1

# show_images_from_npy(shot_noise_5)


# In[4]:


# class Residual(tf.keras.Model):
    # """The Residual block of ResNet."""
    # def __init__(self, num_channels, use_1x1conv=False, strides=1):
        # super().__init__()
        # self.conv1 = tf.keras.layers.Conv2D(
            # num_channels,
            # padding='same',
            # kernel_size=3,
            # strides=strides)
        # self.conv2 = tf.keras.layers.Conv2D(
            # num_channels,
            # kernel_size=3,
            # padding='same')
        # self.conv_1x1 = None
        # if use_1x1conv:
            # self.conv_1x1 = tf.keras.layers.Conv2D(
                # num_channels,
                # kernel_size=1,
                # strides=strides)
        # self.bn1 = tf.keras.layers.BatchNormalization()
        # self.bn2 = tf.keras.layers.BatchNormalization()

    # def call(self, X):
        # Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        # Y = self.bn2(self.conv2(Y))
        # if self.conv_1x1 is not None:
            # X = self.conv_1x1(X)
        # Y += X
        # return tf.keras.activations.relu(Y)

# class ResnetBlock(tf.keras.layers.Layer):
    # def __init__(self, num_channels, num_residuals, downscale=True,
                 # **kwargs):
        # super(ResnetBlock, self).__init__(**kwargs)
        # self.residual_layers = []
        # for i in range(num_residuals):
            # if i == 0 and downscale:
                # self.residual_layers.append(
                    # Residual(num_channels, use_1x1conv=True, strides=2))
            # else:
                # self.residual_layers.append(Residual(num_channels))

    # def call(self, X):
        # for layer in self.residual_layers.layers:
            # X = layer(X)
        # return X

# class ResnetBlock(tf.keras.layers.Layer):
    # def __init__(self, num_channels, num_residuals, downscale=True,
                 # **kwargs):
        # super(ResnetBlock, self).__init__(**kwargs)
        # self.residual_layers = []
        # for i in range(num_residuals):
            # if i == 0 and downscale:
                # self.residual_layers.append(
                    # Residual(num_channels, use_1x1conv=True, strides=2))
            # else:
                # self.residual_layers.append(Residual(num_channels))

    # def call(self, X):
        # for layer in self.residual_layers.layers:
            # X = layer(X)
        # return X

# class ResNet(tf.keras.Model):
    # def __init__(self, num_classes=10, augment=False):
        # super(ResNet, self).__init__()
        # self.augment = augment
        # self.augmentation_block = tf.keras.Sequential(
            # [tf.keras.layers.Rescaling(scale=1.0 / 255),
             # tf.keras.layers.RandomFlip("horizontal_and_vertical"),
             # tf.keras.layers.RandomZoom(
                 # height_factor=(-0.05, -0.15),
                 # width_factor=(-0.05, -0.15)),
             # tf.keras.layers.RandomRotation(0.3)])
        # self.block_a = tf.keras.Sequential(
            # [tf.keras.layers.Conv2D(64,
                                    # kernel_size=7,
                                    # strides=2,
                                    # padding='same'),
             # tf.keras.layers.BatchNormalization(),
             # tf.keras.layers.MaxPool2D(pool_size=3,
                                       # strides=2,
                                       # padding='same')])
        # self.block_b = ResnetBlock(64, 2, downscale=False)
        # self.block_c = ResnetBlock(128, 2)
        # self.block_d = ResnetBlock(256, 2)
        # self.block_e = ResnetBlock(512, 2)
        # self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        # self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax")

    # def call(self, inputs):
        # if self.augment:
            # x = self.augmentation_block(inputs)
            # x = self.block_a(x)
        # else:
            # x = self.block_a(inputs)
        # x = self.block_b(x)
        # x = self.block_c(x)
        # x = self.block_d(x)
        # x = self.block_e(x)
        # x = self.global_pool(x)
        # x = tf.keras.layers.Dropout(0.25)(x)
        # return self.classifier(x)


# In[ ]:


def lr_scheduler(epoch, lr):
    new_lr = lr
    if epoch <= 70:
        pass
    elif epoch == 71 or epoch == 81 or epoch == 91:
        new_lr = lr * 0.1
    return new_lr
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

checkpoint_filepath = './models_cifar/checkpoint_cifar10_lr_red'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

net = ResNet(input_shape=(32, 32, 3), augment=True)
# net.build((1, 32, 32, 3))
# net.summary()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
net.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics="accuracy")
net.fit(cifar10_train, validation_data=cifar10_test, epochs=100, callbacks=[reduce_lr, model_checkpoint_callback])
print("Accuracy on test set: {:.2f}".format(net.evaluate(cifar10_test)[1]))

