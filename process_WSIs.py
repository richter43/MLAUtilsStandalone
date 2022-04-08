import os
import sys
import pickle
from glob import glob

import matplotlib.pyplot as plt
import tensorflow as tf
from utils import seaborn_cm
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
rootdir_wsi = "/space/ponzio/CRC_WSIs/"
rootdir_src = "/space/ponzio/teaching-MLinAPP/src/"
save_dir = os.path.join(rootdir_wsi, "CRC_WSIs_tiles_MC-drop")
try:
    os.makedirs(save_dir)
except FileExistsError:
    print("{} exists.".format(save_dir))
sys.path.append(rootdir_src)
from wsi_utils import DatasetManager


# ----------------------
tile_size = 500
tile_new_size = 112
overlap = 1
epochs = 100
learning_rate = 0.01
batch_size = 128
channels = 3
class_dict = {
    "H": 0,
    "NH": 1
}
checkpoint_filepath = "/space/ponzio/teaching-MLinAPP/models_crc_mc-drop/1_HvsNH.h5"
# ----------------------
input_shape = (tile_new_size, tile_new_size, channels)
num_classes = len(class_dict.keys())
wsi_file_paths_test = glob(os.path.join(rootdir_wsi, '*.svs'))
wsi_labels_numerical_test = [0]*len(wsi_file_paths_test)

model = tf.keras.models.load_model(checkpoint_filepath)

j = 1
for wsi, label in zip(wsi_file_paths_test, wsi_labels_numerical_test):
    print("Processing {} - {}/{}".format(wsi, j, len(wsi_file_paths_test)))
    dataset_manager_test = DatasetManager([wsi],
                                          [label],
                                          tile_size=tile_size,
                                          tile_new_size=tile_new_size,
                                          num_classes=2,
                                          channels=channels,
                                          batch_size=batch_size)
    dataset_test = dataset_manager_test.make_dataset(shuffle=False)
    preds = model.predict(dataset_test)
    tile_placeholders = dataset_manager_test.get_tile_placeholders_filt
    for tile, pred in zip(tile_placeholders, preds):
        tile['prediction'] = pred
    filename = os.path.basename(wsi).split('.')[0]
    filepath = os.path.join(save_dir, "{}_tile_placeholders.pickle".format(filename))
    with open(filepath, "wb") as fp:
        pickle.dump(tile_placeholders, fp)
    j += 1
