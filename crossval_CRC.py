import os
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from wsi_utils import DatasetManager
from sklearn import model_selection
from utils import seaborn_cm
from sklearn.metrics import confusion_matrix
# -------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
rootdir_wsi = "/space/ponzio/CRC_ROIs_4_classes/"
rootdir_src = "/space/ponzio/teaching-MLinAPP/src/"
output_dir = "../models_crc_1000x1000_MC-drop"
checkpoint_filename = "HvsNH.h5"
n_splits = 3
tile_size = 1000
tile_new_size = 112
overlap = 0.5
epochs = 10
learning_rate = 0.01
batch_size = 128
channels = 3
num_classes = 2
class_dict = {
    "AC": 1,
    "AD": 1,
    "H": 0
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

# Make dataset >>>>
wsi_file_paths = glob(os.path.join(rootdir_wsi, '*.svs'))
df = pd.DataFrame([os.path.basename(slide).split('.')[0].split('_') for slide in wsi_file_paths], columns=["Patient",
                                                                                                           "Type",
                                                                                                           "Sub-type",
                                                                                                           "Dysplasia",
                                                                                                           "#-Annotation"])
df['Path'] = wsi_file_paths
# df = df.sample(frac=0.05)
# group_kfold = model_selection.GroupKFold(n_splits=n_splits)
# Make dataset <<<<
fold = 1
group_kfold = model_selection.GroupShuffleSplit(test_size=.35, n_splits=n_splits, random_state=7)
for train_index, test_index in group_kfold.split(df, groups=df['Patient']):
    print('#'*len("Fold {}/{}".format(fold, n_splits)))
    print("Fold {}/{}".format(fold, n_splits))
    print('#'*len("Fold {}/{}".format(fold, n_splits)))
    checkpoint_filepath = os.path.join(output_dir, "{}_".format(fold) + checkpoint_filename)

    train = df.iloc[train_index]
    test = df.iloc[test_index]
    filepath = os.path.join(output_dir, "{}_".format(fold) + "train.csv")
    train.to_csv(filepath)
    filepath = os.path.join(output_dir, "{}_".format(fold) + "test.csv")
    test.to_csv(filepath)
    wsi_file_paths_train = train['Path']
    wsi_labels_categorical_train = list(train['Type'])
    wsi_labels_numerical_train = [class_dict[label] for label in wsi_labels_categorical_train]
    wsi_file_paths_test = test['Path']
    wsi_labels_categorical_test = test['Type']
    wsi_labels_numerical_test = [class_dict[label] for label in wsi_labels_categorical_test]
    print("Unique labels in test: {}".format(set(wsi_labels_categorical_test)))
    print("Unique patients in test: {}".format(set(test['Patient'])))
    print("Unique patients in train: {}".format(set(train['Patient'])))

    dataset_manager_train = DatasetManager(wsi_file_paths_train,
                                           wsi_labels_numerical_train,
                                           tile_size=tile_size,
                                           tile_new_size=tile_new_size,
                                           overlap=overlap,
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
        fig, ax = plt.subplots(10, 10, figsize=(30, 30))
        ax = ax.ravel()
        j = 0
        for image, label in zip(batch_x[:100], batch_y[:100]):
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
    x = tf.keras.layers.Dropout(0.5)(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
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

    y_pred = model.predict(dataset_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.array([tile['label'] for tile in dataset_manager_test.get_tile_placeholders_filt if tile[
        'std'] > dataset_manager_test.std_threshold])
    cm = confusion_matrix(y_true, y_pred)
    mean_acc = np.mean(np.diag(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]))
    print("@" * len("Accuracy: {:.2f}".format(mean_acc)))
    print("Accuracy: {:.2f}".format(mean_acc))
    print("@" * len("Accuracy: {:.2f}".format(mean_acc)))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    seaborn_cm(cm, ax, ["H", "NH"])
    filepath = os.path.join(output_dir, "{}_".format(fold) + "cm.png")
    fig.savefig(filepath)

    fold += 1