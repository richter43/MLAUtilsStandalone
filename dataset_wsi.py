import openslide
from math import floor
import tensorflow as tf
import openslide.deepzoom as dz

class DatasetWSI:
    def __init__(self,
                 wsi_file_paths,
                 wsi_labels,
                 class_dict,
                 channels=3,
                 batch_size=32,
                 tile_size=500,
                 overlap=6):
        self.channels = channels
        self.batch_size = batch_size
        self.num_classes = len(class_dict.keys())
        self.class_dict = class_dict
        self.overlap = overlap
        self.tile_size = tile_size
        self.openSlideObjects = [(openslide.OpenSlide(slide),
                                  label) for slide, label in zip(wsi_file_paths, wsi_labels)]
        self.deepZoomObjects = [(dz.DeepZoomGenerator(slide,
                                                      tile_size=self.tile_size,
                                                      overlap=self.overlap,
                                                      limit_bounds=True),
                                 label) for slide, label in self.openSlideObjects]
        self.deepZoomObjects = [(slide, label, slide.get_tile(10, (0, 0))) for slide, label in self.deepZoomObjects]
        self.tilePlaceholders = [self._create_tile_placeholders(
            slide,
            label,
            slide.level_count - 2,
            preview
        ) for slide, label, preview in self.deepZoomObjects]
        self.tilePlaceholders = [item for sublist in self.tilePlaceholders for item in sublist]
        print(self.tilePlaceholders[0])
        print(self.tilePlaceholders[1])
        print(self.tilePlaceholders[2])
        print("Stats:\n{} slides".format(len(self.deepZoomObjects)))
        print("{} tiles".format(len(self.tilePlaceholders)))
        print("{} classes".format(len(set(wsi_labels))))

    def _create_tile_placeholders(self, deep_zoom_object, label, level, preview):
        if level >= deep_zoom_object.level_count:
            raise(RuntimeError('Requested level is not available'))

        out = []
        preview = tf.reshape(tf.cast(preview.getdata(), dtype=tf.uint8),
                             (preview.size[0], preview.size[1], 3))
        preview = tf.image.convert_image_dtype(preview, dtype=tf.float32)
        preview_scale = 2**(level-10)

        for i in range(deep_zoom_object.level_tiles[level][0]):
            for j in range(deep_zoom_object.level_tiles[level][1]):
                tmp = {
                    'deepZoomObject': deep_zoom_object,
                    'level': level,
                    'coordinates': (i, j),
                    'label': label}
                position = (floor(i*self.tile_size/preview_scale), floor(j*self.tile_size/preview_scale),
                            floor(i + 1*self.tile_size/preview_scale), floor(j + 1*self.tile_size/preview_scale))
                crop = preview[position[0]:position[1], position[2]:position[3]]
                if tf.reduce_mean(tf.math.reduce_std(crop, axis=-1)) > 0.02:
                    out.append(tmp)
        return out

    def _to_image(self, x):
        tile = self.tilePlaceholders[x.numpy()]
        pil_object = tile['deepZoomObject'].get_tile(tile['level'], tile['coordinates'])
        im_size = pil_object.size
        img = tf.reshape(tf.cast(pil_object.getdata(), dtype=tf.uint8), (im_size[0], im_size[1], 3))
        return tf.image.convert_image_dtype(img, dtype=tf.float32), tf.cast(tf.one_hot(self.class_dict[tile['label']],
                                                                                       self.num_classes,
                                                                                       name='label', axis=-1),
                                                                            tf.float32)

    @staticmethod
    def _filter_white(x, label):
        if tf.reduce_mean(tf.math.reduce_std(x, axis=-1)) < 0.02:
            return False
        return True

    def _filter_border(self, x):
        if x.shape[0] != self.tile_size + 2*self.overlap or x.shape[1] != self.tile_size + 2*self.overlap:
            return False
        else:
            return True

    def _fixup_shape(self, image, label):
        """
        Tensor.shape is determined at graph build time (tf.shape(tensor) gets you the runtime shape).
        In tf.numpy_function/tf.py_function “don’t build a graph for this part, just run it in python”.
        So none of the code in such functions runs during graph building, and TensorFlow does not know the shape in there.
        With the function _fixup_shape we set the shape of the tensors.
        """
        image.set_shape([self.tile_size + 2 * self.overlap,
                         self.tile_size + 2 * self.overlap,
                         self.channels])
        label.set_shape([self.num_classes])
        return image, label

    def make_dataset(self, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices([i for i in range(len(self.tilePlaceholders))])
        if shuffle:
            dataset = dataset.shuffle(50000)
        dataset = dataset.map(lambda x: tf.py_function(self._to_image, [x], Tout=[tf.float32, tf.float32]),
                              num_parallel_calls=8)
        dataset = dataset.filter(self._filter_white)
        dataset = dataset.filter(lambda x, label: tf.py_function(self._filter_border, [x], tf.bool))
        dataset = dataset.map(lambda x, y: self._fixup_shape(x, y))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset





