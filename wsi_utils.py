import os
import math
import openslide
import numpy as np
import tensorflow as tf
from matplotlib import cm
from PIL import Image, ImageFilter


class SlideManager:
    def __init__(self, tile_size, overlap=1, verbose=0):
        """
        # SlideManager provides an easy way to generate a cropList object.
        # This object is not tied to a particular slide and can be reused to crop many slides using the same settings.
        """
        self.tile_size = tile_size
        self.level = 0
        self.overlap = int(1/overlap)
        self.verbose = verbose

    def __generateSections__(self,
                             x_start,
                             y_start,
                             width,
                             height,
                             downsample_factor,
                             filepath):
        side = self.tile_size
        step = int(side / self.overlap)
        self.__sections__ = []

        n_tiles = 0
        # N.B. Tiles are considered in the 0 level
        for y in range(int(math.floor(height / step))):
            for x in range(int(math.floor(width / step))):
                # x * step + side is right margin of the given tile
                if x * step + side > width or y * step + side > height:
                    continue
                n_tiles += 1
                self.__sections__.append(
                    {'top': y_start + step * y, 'left': x_start + step * x,
                     'size': math.floor(side / downsample_factor)})
        if self.verbose:
            print("-"*len("{} stats:".format(filepath)))
            print("{} stats:".format(filepath))
            print("step: {}".format(step))
            print("y: {}".format(y_start))
            print("x: {}".format(x_start))
            print("slide width {}".format(width))
            print("slide height {}".format(height))
            print("downsample factor: {}".format(downsample_factor))
            print("# of tiles:{}".format(n_tiles))
            print("-" * len("{} stats:".format(filepath)))

    def crop(self, filepath_slide, label=None):
        self.level = 0
        slide = openslide.OpenSlide(filepath_slide)
        downsample = slide.level_downsamples[self.level]
        if 'openslide.bounds-width' in slide.properties.keys():
            bounds_width = int(slide.properties['openslide.bounds-width'])
            bounds_height = int(slide.properties['openslide.bounds-height'])
            bounds_x = int(slide.properties['openslide.bounds-x'])
            bounds_y = int(slide.properties['openslide.bounds-y'])
        else:
            bounds_width = slide.dimensions[0]
            bounds_height = slide.dimensions[1]
            bounds_x = 0
            bounds_y = 0

        self.__generateSections__(bounds_x,
                                  bounds_y,
                                  bounds_width,
                                  bounds_height,
                                  downsample,
                                  filepath_slide)
        indexes = self.__sections__
        for index in indexes:
            index['filepath_slide'] = filepath_slide
            index['level'] = self.level
            index['label'] = label
        return indexes


class DatasetManager:
    def __init__(self,
                 filepaths,
                 labels,
                 tile_size,
                 tile_new_size=None,
                 num_classes=None,
                 overlap=1,
                 channels=3,
                 batch_size=32,
                 one_hot=True,
                 std_threshold=20,
                 verbose=0):
        if tile_new_size:
            self.new_size = tile_new_size
        else:
            self.new_size = tile_size
        self.crop_size = tile_size
        self.one_hot = one_hot
        self.overlap = overlap
        self.std_threshold = std_threshold
        if num_classes is None:
            self.num_classes = len(set(labels))
        else:
            self.num_classes = num_classes
        self.channels = channels
        self.batch_size = batch_size
        self.section_manager = SlideManager(tile_size, overlap=self.overlap, verbose=verbose)
        self.tile_placeholders = sum([self.section_manager.crop(
            filepath,
            label=label) for filepath, label in zip(filepaths, labels)], [])
        print("*"*len("Found in total {} tiles.".format(len(self.tile_placeholders))))
        print("Found in total:\n {} tiles\n belonging to {} slides".format(len(self.tile_placeholders),
                                                                           len(filepaths)))
        print("*" * len("Found in total {} tiles.".format(len(self.tile_placeholders))))

    def _to_image(self, x):
        slide = openslide.OpenSlide(self.tile_placeholders[x.numpy()]['filepath_slide'])
        pil_object = slide.read_region([self.tile_placeholders[x.numpy()]['left'],
                                        self.tile_placeholders[x.numpy()]['top']],
                                       self.tile_placeholders[x.numpy()]['level'],
                                       [self.tile_placeholders[x.numpy()]['size'],
                                        self.tile_placeholders[x.numpy()]['size']])
        pil_object = pil_object.convert('RGB')
        pil_object = pil_object.resize(size=(self.new_size, self.new_size))
        self.tile_placeholders[x.numpy()]['std'] = np.std(np.array(pil_object))
        label = self.tile_placeholders[x.numpy()]['label']
        im_size = pil_object.size
        img = tf.reshape(tf.cast(pil_object.getdata(), dtype=tf.uint8), (im_size[0], im_size[1], 3))
        if self.tile_placeholders[x.numpy()]['std'] > self.std_threshold:
            return tf.image.convert_image_dtype(img, dtype=tf.float32), tf.cast(label, tf.float32)
        else:
            return tf.image.convert_image_dtype(img, dtype=tf.float32), tf.cast(-1, tf.float32)

    @staticmethod
    def _filter_white(x, label):
        if tf.math.equal(label, -1):
            return False
        return True

    def _to_one_hot(self, image, label):
        return image, tf.cast(tf.one_hot(tf.cast(label, tf.int32),
                                         self.num_classes,
                                         name='label', axis=-1),
                              tf.float32)

    def _fixup_shape(self, image, label):
        """
        Tensor.shape is determined at graph build time (tf.shape(tensor) gets you the runtime shape).
        In tf.numpy_function/tf.py_function “don’t build a graph for this part, just run it in python”.
        So none of the code in such functions runs during graph building, and TensorFlow does not know the shape in there.
        With the function _fixup_shape we set the shape of the tensors.
        """
        image.set_shape([self.new_size,
                         self.new_size,
                         self.channels])
        if self.one_hot:
            label.set_shape([self.num_classes])
        else:
            label.set_shape([])
        return image, label

    def make_dataset(self, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices([i for i in range(len(self.tile_placeholders))])
        if shuffle:
            dataset = dataset.shuffle(50000)
        dataset = dataset.map(lambda x: tf.py_function(self._to_image, [x], Tout=[tf.float32, tf.float32]),
                              num_parallel_calls=8)
        dataset = dataset.filter(self._filter_white)
        if self.one_hot:
            dataset = dataset.map(self._to_one_hot)
        dataset = dataset.map(lambda x, y: self._fixup_shape(x, y))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    @property
    def get_tile_placeholders(self):
        return self.tile_placeholders

    @property
    def get_tile_placeholders_filt(self):
        return list(filter(lambda x: x["std"] > self.std_threshold, self.tile_placeholders))


def get_heatmap(tile_placeholders,
                class_to_map,
                num_classes,
                level_downsample,
                threshold=0.5,
                tile_placeholders_mapping_key='prediction',
                colormap=cm.get_cmap('Blues')):

    """
    Builds a 3 channel map.
    The first three channels represent the sum of all the probabilities of the crops which contain that pixel
    belonging to classes 0-1, the fourth hold the number of crops which contain it.
    """

    slide = openslide.OpenSlide(tile_placeholders[0]['filepath_slide'])

    if 'openslide.bounds-width' in slide.properties.keys():
        # Here to consider only the rectangle bounding the non-empty region of the slide, if available.
        # These properties are in the level 0 reference frame.
        bounds_width = int(slide.properties['openslide.bounds-width'])
        bounds_height = int(slide.properties['openslide.bounds-height'])
        bounds_x = int(slide.properties['openslide.bounds-x'])
        bounds_y = int(slide.properties['openslide.bounds-y'])

        region_lv0 = (bounds_x,
                      bounds_y,
                      bounds_width,
                      bounds_height)
    else:
        # If bounding box of the non-empty region of the slide is not available
        # Slide dimensions of given level reported to level 0
        region_lv0 = (0, 0, slide.level_dimensions[0][0], slide.level_dimensions[0][1])

    region_lv0 = [round(x) for x in region_lv0]
    region_lv_selected = [round(x * level_downsample) for x in region_lv0]
    probabilities = np.zeros((region_lv_selected[3], region_lv_selected[2], 3))
    for tile in tile_placeholders:
        top = math.ceil(tile['top'] * level_downsample)
        left = math.ceil(tile['left'] * level_downsample)
        side = math.ceil(tile['size'] * level_downsample)
        top -= region_lv_selected[1]
        left -= region_lv_selected[0]
        side_x = side
        side_y = side
        if top < 0:
            side_y += top
            top = 0
        if left < 0:
            side_x += left
            left = 0
        if side_x > 0 and side_y > 0:
            probabilities[top:top + side_y, left:left + side_x, 0:num_classes] = np.array(
                tile[tile_placeholders_mapping_key][class_to_map])

    probabilities = probabilities * 255
    probabilities = probabilities.astype('uint8')

    map_ = probabilities[:, :, class_to_map]
    map_ = Image.fromarray(map_).filter(ImageFilter.GaussianBlur(3))
    map_ = np.array(map_) / 255
    map_[map_ < threshold] = 0
    segmentation = (map_ * 255).astype('uint8')
    map_ = colormap(np.array(map_))
    roi_map = Image.fromarray((map_ * 255).astype('uint8'))
    roi_map.putalpha(75)

    slide_image = slide.get_thumbnail((region_lv_selected[2], region_lv_selected[3]))
    slide_image = slide_image.convert('RGBA')
    slide_image.alpha_composite(roi_map)
    slide_image.convert('RGBA')
    # segmentation[segmentation != 0] = 255
    return slide_image, segmentation
