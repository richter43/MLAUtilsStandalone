import math
import openslide
import tensorflow as tf


class SlideManager:
    def __init__(self, tile_size, overlap=1, verbose=0):
        """
        # SlideManager provides an easy way to generate a cropList object.
        # This object is not tied to a particular slide and can be reused to crop many slides using the same settings.
        @param tile_size: crop_size
        @param overlap: overlap (%)
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
                 channels=3,
                 batch_size=32,
                 verbose=0):
        if tile_new_size:
            self.new_size = tile_new_size
        else:
            self.new_size = tile_size
        self.crop_size = tile_size
        self.num_classes = len(set(labels))
        self.channels = channels
        self.batch_size = batch_size
        self.section_manager = SlideManager(tile_size, overlap=1, verbose=verbose)
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
        label = self.tile_placeholders[x.numpy()]['label']
        im_size = pil_object.size
        img = tf.reshape(tf.cast(pil_object.getdata(), dtype=tf.uint8), (im_size[0], im_size[1], 3))
        return tf.image.convert_image_dtype(img, dtype=tf.float32), tf.cast(tf.one_hot(label,
                                                                                       self.num_classes,
                                                                                       name='label', axis=-1),
                                                                            tf.float32)

    def _filter_image(self, x):
        slide = openslide.OpenSlide(self.tile_placeholders[x.numpy()]['filepath_slide'])
        pil_object = slide.read_region([self.tile_placeholders[x.numpy()]['left'],
                                        self.tile_placeholders[x.numpy()]['top']],
                                       self.tile_placeholders[x.numpy()]['level'],
                                       [self.tile_placeholders[x.numpy()]['size'],
                                        self.tile_placeholders[x.numpy()]['size']])
        pil_object = pil_object.convert('RGB')
        if pil_object.thumbnail((1, 1)).getpixel(0, 0) < 0.02:
            self.tile_placeholders[x.numpy()]['removed'] = True
            return True
        else:
            self.tile_placeholders[x.numpy()]['removed'] = False
            return False

    def py_function_filter(self, x):
        return tf.py_function(self._filter_image, x, tf.bool)

    @staticmethod
    def _filter_white(x, label):
        if tf.reduce_mean(tf.math.reduce_std(x, axis=-1)) < 0.02:
            return False
        return True

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
        label.set_shape([self.num_classes])
        return image, label

    def make_dataset(self, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices([i for i in range(len(self.tile_placeholders))])
        if shuffle:
            dataset = dataset.shuffle(50000)
        dataset = dataset.filter(self.py_function_filter)
        dataset = dataset.map(lambda x: tf.py_function(self._to_image, [x], Tout=[tf.float32, tf.float32]),
                              num_parallel_calls=8)
        # dataset = dataset.filter(self._filter_white)
        dataset = dataset.map(lambda x, y: self._fixup_shape(x, y))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    # @property
    # def get_tile_placeholders(self):
    #     dataset = tf.data.Dataset.from_tensor_slices((
    #         [tile_placeholder for tile_placeholder in self.tile_placeholders],
    #         [i for i in range(len(self.tile_placeholders))]
    #     ))
    #
    #     dataset = dataset.filter(self._filter_white)
    #     return dataset
























