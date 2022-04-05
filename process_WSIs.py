import os
import sys
import pickle
from glob import glob
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
rootdir_wsi = "/space/ponzio/CRC_WSIs/"
rootdir_src = "/space/ponzio/teaching-MLinAPP/src/"
sys.path.append(rootdir_src)
from wsi_utils import DatasetManager
from wsi_utils import get_heatmap
import matplotlib

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
checkpoint_filepath = "/space/ponzio/teaching-MLinAPP/models_crc_5-04/1_HvsNH.h5"
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
    filepath = os.path.join(rootdir_wsi, "{}_tile_placeholders.pickle".format(filename))
    with open(filepath, "wb") as fp:
        pickle.dump(tile_placeholders, fp)

    heatmap_NH = get_heatmap(tile_placeholders,
                             class_to_map=1,
                             level_downsample=1/50,
                             num_classes=2,
                             colormap=matplotlib.cm.get_cmap('Reds'),
                             tile_placeholders_mapping_key='prediction')
    heatmap_NH.save(os.path.join(rootdir_wsi, "{}-NH.png".format(filename)))
    heatmap_H = get_heatmap(tile_placeholders,
                            class_to_map=0,
                            level_downsample=1/50,
                            num_classes=2,
                            colormap=matplotlib.cm.get_cmap('Blues'),
                            tile_placeholders_mapping_key='prediction')
    heatmap_H.save(os.path.join(rootdir_wsi, "{}-H.png".format(filename)))
    j += 1
