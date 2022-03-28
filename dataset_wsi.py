from math import floor


class DatasetWSI:
    def __init__(self,
                 dir_slide,
                 class_dict,
                 tile_size=512,
                 overlap=6,
                 format_slide='*.svs'):
        self.files = glob(os.path.join(dir_slide, format_slide))
        self.class_dict = class_dict
        self.overlap = overlap
        self.tile_size = tile_size
        self.openSlideObjects = [(openslide.OpenSlide(slide), os.path.basename(slide).split('_')[1]) for slide in files]
        self.deepZoomObjects = [(dz.DeepZoomGenerator(slide,
                                                      tile_size=self.tile_size,
                                                      overlap=self.overlap,
                                                      limit_bounds=True), label) for slide, label in openSlideObjects]
        self.deepZoomObjects = [(slide, label, slide.get_tile(10,(0,0))) for slide, label in self.deepZoomObjects]
        self.tilePlaceholders = [self._create_tile_placeholders(slide,
                                                                label,
                                                                slide.level_count - 2,
                                                                preview) for slide, label, preview in deepZoomObjects]
        self.tilePlaceholders = [item for sublist in self.tilePlaceholders for item in sublist]

    def _create_tile_placeholders(self, deep_zoom_object, label, level, preview):
        if level>=deep_zoom_object.level_count:
            raise(RuntimeError('Requested level is not available'))

        out=[]
        preview = tf.reshape(tf.cast(preview.getdata(),dtype=tf.uint8),(preview.size[0],preview.size[1],3))
        preview = tf.image.convert_image_dtype(preview,dtype=tf.float32)
        preview_scale = 2**(level-10)

        for i in range(deep_zoom_object.level_tiles[level][0]):
            for j in range(deep_zoom_object.level_tiles[level][1]):
                tmp={
                    'deepZoomObject':deep_zoom_object,
                    'level':level,
                    'coordinates':(i,j),
                    'label':label}
                position = (floor(i*TILE_SIZE/preview_scale), floor(j*TILE_SIZE/preview_scale),
                            floor(i + 1*TILE_SIZE/preview_scale), floor(j + 1*TILE_SIZE/preview_scale))
                crop = preview[position[0]:position[1],position[2]:position[3]]
                if tf.reduce_mean(tf.math.reduce_std(crop,axis=-1))>0.02:
                    out.append(tmp)
        return out

    @staticmethod
    def _to_image(x):
        tile = tilePlaceholders[x.numpy()]
        pil_object = tile['deepZoomObject'].get_tile(tile['level'], tile['coordinates'])
        im_size = pil_object.size
        img = tf.reshape(tf.cast(pil_object.getdata(),dtype=tf.uint8),(im_size[0],im_size[1],3))
        return tf.image.convert_image_dtype(img, dtype=tf.float32), tf.cast(tf.one_hot(classDict[tile['label']], 2, name='label', axis=-1), tf.float32)

    @staticmethod
    def filter_white(x, label):
        if tf.reduce_mean(tf.math.reduce_std(x,axis=-1)) < 0.02:
            #tf.print('Skipped (white)')
            return False
        return True

    @staticmethod
    def filter_border(x):
        if x.shape[0] != TILE_SIZE + 2*OVERLAP or x.shape[1] != TILE_SIZE + 2*OVERLAP:
            #tf.print('Skipped (border) %d - %d'%(x.shape[0],x.shape[1]))
            return False
        else:
            return True

    @staticmethod
    def _fixup_shape(image, label, num_classes):
        image.set_shape([INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
        label.set_shape([NUM_CLASSES])
        return image, label

    @staticmethod
    def make_dataset():
        dataset = tf.data.Dataset.from_tensor_slices([i for i in range(len(tilePlaceholders))])
        dataset = dataset.shuffle(50000)
        dataset = dataset.map(lambda x: tf.py_function(toImage, [x], Tout=[tf.float32, tf.float32]), num_parallel_calls=8)
        dataset = dataset.filter(filterWhite)
        dataset = dataset.filter(lambda x, label: tf.py_function(filterBorder, [x], tf.bool))
        dataset = dataset.map(lambda x, y: _fixup_shape(x, y, num_classes=NUM_CLASSES))
        dataset = dataset.batch(32)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset