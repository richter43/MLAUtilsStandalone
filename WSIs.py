#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[ ]:


# !apt update && apt install -y openslide-tools
# !pip install openslide-python


# In[ ]:


# !rm -r "/content/drive/MyDrive/Teaching&Thesis/Teaching_dataset/teaching-MLinAPP"
# !git clone https://github.com/frpnz/teaching-MLinAPP.git "/content/drive/MyDrive/Teaching&Thesis/Teaching_dataset/teaching-MLinAPP"


# In[ ]:


import os
import sys
import openslide
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
import matplotlib.pyplot as plt
import openslide.deepzoom as dz
from sklearn import model_selection
rootdir_wsi = "/space/ponzio/CRC_ROIs_4_classes/"
rootdir_src = "/space/ponzio/teaching-MLinAPP/src/"
sys.path.append(rootdir_src)
from resnet import ResNet
from dataset_wsi import DatasetWSI
# ----------------------
tile_size = 300
overlap = 6
epochs = 100
learning_rate = 0.001
batch_size = 64
class_dict = {
    "AC": 0,
    "H": 2
}
checkpoint_filepath = './models_crc/checkpoint_crc_2_cls'
# ----------------------
num_classes = len(class_dict.keys())
wsi_file_paths = glob(os.path.join(rootdir_wsi, '*.svs'))
df = pd.DataFrame([os.path.basename(slide).split('.')[0].split('_') for slide in wsi_file_paths], columns=["Patient",
                                                                                                           "Type",
                                                                                                           "Sub-type",
                                                                                                           "Dysplasia",
                                                                                                           "#-Annotation"])
df['Path'] = wsi_file_paths
wsi_file_paths = df['Path']
wsi_labels = df['Type']

print("Train")
dataset_wsi = DatasetWSI(wsi_file_paths,
                         wsi_labels,
                         class_dict,
                         batch_size=batch_size,
                         tile_size=tile_size,
                         overlap=6).make_dataset()
for batch_x, batch_y in dataset_wsi.take(1):
    input_shape = batch_x[0].shape

inv_class_dict = {v: k for k, v in class_dict.items()}
for batch_x, batch_y in dataset_wsi.take(2):
    fig, ax = plt.subplots(5, 5, figsize=(18, 18))
    ax = ax.ravel()
    j = 0
    for image, label in zip(batch_x[:25], batch_y[:25]):
        label = label.numpy()
        img = image.numpy()
        input_shape = img.shape
        ax[j].imshow(img)
        ax[j].axis('off')
        ax[j].set_title("Class: {} - {}".format(inv_class_dict[int(np.argmax(label))], label))
        j += 1
fig.savefig("images.pdf")


data_dir = "../crc_images/train"
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    label_mode="categorical",
    seed=123,
    image_size=(input_shape[0], input_shape[1]),
    batch_size=64)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    label_mode="categorical",
    seed=123,
    image_size=(input_shape[0], input_shape[1]),
    batch_size=64)

data_dir = "../crc_images/test"
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    label_mode="categorical",
    seed=123,
    image_size=(input_shape[0], input_shape[1]),
    batch_size=64)

augmentation_block = [
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomZoom(
        height_factor=(-0.05, -0.15),
        width_factor=(-0.05, -0.15)),
    tf.keras.layers.RandomRotation(0.3),
]
inputs = tf.keras.Input(input_shape)
x =  tf.keras.applications.resnet50.preprocess_input(inputs)
x = tf.keras.layers.Resizing(64, 64)(x)
for layer in augmentation_block:
    x = layer(x, training=False)
base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet")
for j, layer in enumerate(base_model.layers[:100]):
    layer.trainable = False
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.models.Model(inputs=inputs, outputs=x)

model.summary()

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.1,
    patience=7,
    verbose=0,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0,
)

early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.001,
    patience=10,
    verbose=0,
    mode="auto",
    restore_best_weights=True,
)

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
loss = tf.keras.losses.categorical_crossentropy
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

model.fit(train_ds, validation_data=train_ds,
          epochs=epochs, callbacks=[checkpoint_callback, lr_callback, early_stop_callback])

results = model.evaluate(test_ds)
print("Accuracy on test: {}".format(results[1]))
results = model.evaluate(dataset_wsi)
print("Accuracy on WSIs: {}".format(results[1]))

