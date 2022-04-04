import os
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from wsi_utils import DatasetManager
from sklearn import model_selection

# -------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
rootdir_wsi = "/space/ponzio/CRC_ROIs_4_classes/"
rootdir_src = "/space/ponzio/teaching-MLinAPP/src/"
output_dir = "../models_crc"
checkpoint_filename = "HvsNH.h5"
n_splits = 3
tile_size = 1000
tile_new_size = 112
overlap = 1
epochs = 10
learning_rate = 0.01
batch_size = 64
channels = 3
class_dict = {
    "AC": 0,
    "AD": 0,
    "H": 1
}
plot_class_dict = {
    0: "H",
    1: "NH"
}
# -------------------------------------------------
try:
    os.makedirs(output_dir)
except FileExistsError:
    print("{} exists.".format(output_dir))
input_shape = (tile_new_size, tile_new_size, channels)
num_classes = len(set(class_dict.values()))

# Make dataset >>>>
wsi_file_paths = glob(os.path.join(rootdir_wsi, '*.svs'))
df = pd.DataFrame([os.path.basename(slide).split('.')[0].split('_') for slide in wsi_file_paths], columns=["Patient",
                                                                                                           "Type",
                                                                                                           "Sub-type",
                                                                                                           "Dysplasia",
                                                                                                           "#-Annotation"])
df['Path'] = wsi_file_paths
splitter = model_selection.GroupShuffleSplit(test_size=.4, n_splits=n_splits, random_state=7)
split = splitter.split(df, groups=df['Patient'])
# Make dataset <<<<
fold = 1
for train_inds, test_inds in split:
    print('#'*len("Fold {}/{}".format(fold, n_splits)))
    print("Fold {}/{}".format(fold, n_splits))
    print('#'*len("Fold {}/{}".format(fold, n_splits)))
    checkpoint_filepath = os.path.join(output_dir, "{}_".format(fold) + checkpoint_filename)
    train = df.iloc[train_inds]
    test = df.iloc[test_inds]
    wsi_file_paths_train = train['Path']
    wsi_labels_categorical_train = train['Type']
    wsi_labels_numerical_train = [class_dict[label] for label in wsi_labels_categorical_train]
    wsi_file_paths_test = test['Path']
    wsi_labels_categorical_test = test['Type']
    wsi_labels_numerical_test = [class_dict[label] for label in wsi_labels_categorical_test]

    dataset_manager_train = DatasetManager(wsi_file_paths_train,
                                           wsi_labels_numerical_train,
                                           tile_size=tile_size,
                                           tile_new_size=tile_new_size,
                                           channels=channels,
                                           batch_size=batch_size)
    dataset_train = dataset_manager_train.make_dataset()

    dataset_manager_test = DatasetManager(wsi_file_paths_test,
                                          wsi_labels_numerical_test,
                                          tile_size=tile_size,
                                          tile_new_size=tile_new_size,
                                          channels=channels,
                                          batch_size=batch_size)
    dataset_test = dataset_manager_test.make_dataset(shuffle=False)

    for batch_x, batch_y in dataset_train.take(1):
        fig, ax = plt.subplots(4, 4, figsize=(15, 15))
        ax = ax.ravel()
        j = 0
        for image, label in zip(batch_x[:16], batch_y[:16]):
            label = label.numpy()
            img = image.numpy()
            input_shape = img.shape
            ax[j].imshow(img)
            ax[j].axis('off')
            ax[j].set_title("{}\n{}\nshape:{}".format(plot_class_dict[int(np.argmax(label))], label, image.shape))
            j += 1
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "{}_train_images.pdf".format(fold)))

    for batch_x, batch_y in dataset_test.take(1):
        fig, ax = plt.subplots(4, 4, figsize=(15, 15))
        ax = ax.ravel()
        j = 0
        for image, label in zip(batch_x[:100], batch_y[:16]):
            label = label.numpy()
            img = image.numpy()
            input_shape = img.shape
            ax[j].imshow(img)
            ax[j].axis('off')
            ax[j].set_title("Class: {} - {}\nshape:{}".format(plot_class_dict[int(np.argmax(label))], label, image.shape))
            j += 1
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "{}_test_images.pdf".format(fold)))

    # CNN model >>>>
    augmentation_block = [
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.05, -0.15),
            width_factor=(-0.05, -0.15)),
        tf.keras.layers.RandomRotation(0.3),
    ]
    inputs = tf.keras.Input(input_shape)
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    for layer in augmentation_block:
        x = layer(x, training=False)
    base_model = tf.keras.applications.ResNet50V2(include_top=False, weights="imagenet")
    for j, layer in enumerate(base_model.layers[:100]):
        layer.trainable = False
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    # CNN model <<<<

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='accuracy',
        mode='max',
        save_best_only=True)

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='accuracy',
        factor=0.1,
        patience=5,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
    )

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="accuracy",
        min_delta=0.001,
        patience=15,
        verbose=0,
        mode="auto",
        restore_best_weights=True,
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss = tf.keras.losses.categorical_crossentropy
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.fit(dataset_train, epochs=epochs, callbacks=[checkpoint_callback, lr_callback, early_stop_callback])
    print("Loading {}".format(checkpoint_filepath))
    model = tf.keras.models.load_model(checkpoint_filepath)
    results = model.evaluate(dataset_test, verbose=0)
    print("@"*len("Accuracy: {:.2f}".format(results[1])))
    print("Accuracy: {:.2f}".format(results[1]))
    print("@" * len("Accuracy: {:.2f}".format(results[1])))
    fold += 1