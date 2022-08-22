import math
import os
from typing import List, Optional, Generator

import threading
import itertools
import pathos
import matplotlib.pyplot as plt
import numpy as np
import openslide
import tensorflow as tf
import torch
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
import logging
import time

from .ancillary_definitions import RenalCancerType
from .annotation_utils_asap import get_points_xml_asap, get_region_lv0
from .annotation_utils_dataclasses import PointInfo
from .wsi_utils_dataclasses import Section, SlideMetadata
from .utils import UtilException, get_label_from_path, image_entropy, CropType, slide_read_region
from .log_utils import LOGGER_NAME


class WSIDatasetTorch(Dataset):

    def __init__(self, section_list: List[Section], crop_size: int, one_hot: bool = True, annotated_only: bool = False):
        super(WSIDatasetTorch, self).__init__()
        self.section_list : List[Section] = section_list
        self.crop_size = crop_size
        self.num_classes = len(RenalCancerType)
        self.one_hot = one_hot

    def __len__(self):
        return len(self.section_list)

    def __getitem__(self, index):

        section = self.section_list[index]

        slide = openslide.OpenSlide(section.wsi_path)
        pil_object = slide_read_region(slide, section.x, section.y, section.level, section.size)
        pil_object = pil_object.resize(size=(self.crop_size, self.crop_size))
        
        torch_tensor_convertor = transforms.ToTensor()
        img = torch_tensor_convertor(pil_object)

        #Patch can never be white, label is always accurate
        label = torch.tensor(section.label)

        if self.one_hot and label != -1:
            return img, F.one_hot(label, num_classes=self.num_classes)
        else: 
            return img, label

class SlideManager:
    def __init__(self, tile_size: int, overlap: bool = True, verbose: bool = False, remove_low_information=True, information_threshold=40.0):
        """
        # SlideManager provides an easy way to generate a cropList object.
        # This object is not tied to a particular slide and can be reused to crop many slides using the same settings.
        """
        self.tile_size = tile_size
        self.level = 0
        self.overlap = int(1/overlap)
        self.verbose = verbose
        self.remove_low_information = remove_low_information
        self.information_threshold = information_threshold
        self.__sections: List[Section]= []
        self.logger = logging.getLogger(LOGGER_NAME)

    # The usage of an encapsulating dunder doesn't seem to fit the use case here.
    def __generate_sections(self,
                             x_start: int,
                             y_start: int,
                             width: int,
                             height: int,
                             downsample_factor: float,
                             filepath: str, 
                             slide: openslide.OpenSlide):
        side = self.tile_size
        step = side // self.overlap
        # The usage of an encapsulating dunder doesn't seem to fit the use case here.
        self.__sections = []

        # N.B. Tiles are considered in the 0 level
        # Attempted to optimize this section with processor and threads, their lifetime is way too small to justify the locks 
        label = get_label_from_path(filepath)

        n_tiles = self.__create_sections(x_start, x_start + width - side, y_start, y_start + height - side, step, side, downsample_factor, filepath, label, slide)

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

    def __generate_sections_xml(self,
                                filepath: str,
                                point_info: PointInfo):

        slide = openslide.OpenSlide(filepath)
        downsample_factor = slide.level_downsamples[self.level]

        _ , (bounds_x, bounds_y, bounds_width, bounds_height) = get_region_lv0(slide)

        side = self.tile_size
        step = side // self.overlap
        # The usage of an encapsulating dunder doesn't seem to fit the use case here.
        self.__sections = []

        n_tiles = 0
        bounds_x_final = bounds_x + bounds_width
        bounds_y_final = bounds_y + bounds_height
        # N.B. Tiles are considered in the 0 level
        # Attempted to optimize this section with processor and threads, their lifetime is way too small to justify the locks

        for ann_data in point_info.ann_list: 
            x_init, y_init, x_final, y_final = ann_data.polygon.bounds

            label = ann_data.group_name

            #NOTE: Overlayed means that the crops are extracted from a larger, overlapping rectangle (larger by size of side)
            overlayed_x_init = bounds_x if (x_init - side) < bounds_x else x_init - side
            overlayed_y_init = bounds_y if (y_init - side) < bounds_y else y_init - side
            overlayed_x_final = bounds_x_final if (x_final + side) > bounds_x_final else x_final + side
            overlayed_y_final = bounds_y_final if (y_final + side) > bounds_y_final else y_final + side

            n_tiles += self.__create_sections(int(overlayed_x_init), int(overlayed_x_final), int(overlayed_y_init), int(overlayed_y_final), step, side, downsample_factor, filepath, label, slide)

        if self.verbose:
            print("-"*len("{} stats:".format(filepath)))
            print("{} stats:".format(filepath))
            print("step: {}".format(step))
            print("y: {}".format(bounds_y))
            print("x: {}".format(bounds_x))
            print("slide width {}".format(bounds_width))
            print("slide height {}".format(height))
            print("downsample factor: {}".format(downsample_factor))
            print("# of tiles:{}".format(n_tiles))
            print("-" * len("{} stats:".format(filepath)))

    def __create_sections(self, x_init: int, x_final: int, y_init: int, y_final, step: int, side: int, downsample_factor: float, filepath: str, label: int, slide: openslide.OpenSlide):

        n_tiles = 0

        for y, x in itertools.product(range(y_init, y_final, step), range(x_init, x_final, step)):
            n_tiles += 1
            s = Section(x=x, y=y, size=int(side // downsample_factor), level=self.level, wsi_path=filepath, label=label)
            if self.remove_low_information:
                # self.logger.debug(f"Reading: section: {s}\nSlide:{slide}")
                information = image_entropy(slide, s)
                # self.logger.debug(f"Amount of information: {information}")
                if information > self.information_threshold:
                    s.std = information
                    self.__sections.append(s)
            else:
                self.__sections.append(s)
        return n_tiles

    def __crop_xml(self, slide_metadata: SlideMetadata, annotated_only: bool) -> List[Section]:

        if annotated_only:
            point_info = PointInfo(get_points_xml_asap(slide_metadata.annotation_path))
        else:
            point_info = PointInfo(get_points_xml_asap(slide_metadata.annotation_path, "tumor"))

        self.__generate_sections_xml(slide_metadata.wsi_path, point_info)

        patches_to_drop_i = []

        for i, section in enumerate(self.__sections):
            label = -1
            if section.label == "tumor":
                # Assigns label depending on the intersection of the image with the annotation
                section.wsi_path = slide_metadata.wsi_path
                square = section.square
                intercepting_polygons = point_info.strtree.query(square)
                
                for poly in intercepting_polygons:
                    intersected_area = poly.intersection(square).area
                    large_intersection_bool = intersected_area / square.area >= 0.5

                    if large_intersection_bool:
                        label = slide_metadata.label
                        break
            else:
                label = RenalCancerType.NOT_CANCER.value

            if not annotated_only:
                #should be backward compatible
                section.label = label if label !=-1 else RenalCancerType.NOT_CANCER.value
            else:
                if label != -1:
                    #proceed normally
                    section.label = label
                else:
                    #mark the patch as dropped, it's not annotated and we don't want it
                    patches_to_drop_i.append(i)

        if annotated_only:
            self.logger.debug(f"Dropping {len(patches_to_drop_i)} non-annotated patches")
            for dropped, i_to_drop in enumerate(patches_to_drop_i):
                self.__sections.pop(i_to_drop-dropped)

        return self.__sections

    def __crop_roi(self, slide_metadata: SlideMetadata) -> List[Section]:

        slide = openslide.OpenSlide(slide_metadata.annotation_path)
        downsample = slide.level_downsamples[self.level]

        _ , (bounds_x, bounds_y, bounds_width, bounds_height) = get_region_lv0(slide)

        self.__generate_sections(bounds_x,
                                  bounds_y,
                                  bounds_width,
                                  bounds_height,
                                  downsample,
                                  slide_metadata.annotation_path, 
                                  slide)

        return self.__sections

    def crop(self, slide_metadata: SlideMetadata, annotated_only: bool = False) -> List[Section]:
        """Crops 

        Args:
            slide_metadata (SlideMetadata): _description_
            annotated_only (bool): _description_

        Returns:
            List[Section]: _description_
        """

        if slide_metadata.is_roi:
            self.logger.debug("Cropping ROI")
            return self.__crop_roi(slide_metadata)
        else:
            self.logger.debug("Cropping XML")
            return self.__crop_xml(slide_metadata, annotated_only)

def pool_fn(args):
    section_manager, slide_metadata, annotated_only = args
    try:
        return section_manager.crop(slide_metadata, annotated_only=annotated_only)
    except UtilException:
        return None

class DatasetManager:
    def __init__(self,
                 inputs: Generator[SlideMetadata, SlideMetadata, None],
                 tile_size: int,
                 overlap: float = 1.0,
                 channels: int = 3,
                 batch_size: int = 32,
                 one_hot: bool = True,
                 information_threshold: float = 40.0,
                 remove_low_information: bool = True,
                 annotated_only: bool = False,
                 verbose: bool = False,
                 crop_type: CropType = CropType.pool_threading):
        """_summary_

        Args:
            inputs (List[SlideMetadata]): List of objects that contain information about the location of the slide, annotation or any other metadata
            tile_size (int): Resulting image will be tile_size x tile_size
            overlap (float, optional): Ratio of which images will be overlapped between each other. Defaults to 1.0 .
            channels (int, optional): Number of channels of the image. Defaults to 3.
            batch_size (int, optional): Batch size of the dataset. Defaults to 32.
            one_hot (bool, optional): Encoded to one hot. Defaults to True.
            std_threshold (int, optional): _description_. Defaults to 20.
            verbose (bool, optional): Prints further information about the execution. Defaults to False.
            crop_type (CropType, optional): The action of cropping and computing amount of information is done either in a serial manner or in parallel. Defaults to CropType.pool_threading.
        """

        self.crop_size = tile_size
        self.one_hot = one_hot
        self.overlap = overlap
        self.information_threshold = information_threshold
        self.num_classes = len(RenalCancerType)
        self.channels = channels
        self.batch_size = batch_size
        self.annotated_only = annotated_only
        self.section_manager = SlideManager(tile_size, overlap=self.overlap, verbose=verbose, remove_low_information=remove_low_information, information_threshold=information_threshold)
        self.verbose = verbose
        self.tile_placeholders = []
        self.crop_type = crop_type
        self.logger = logging.getLogger(LOGGER_NAME)

        #NOTE: it is recommended to use either multiprocessing or multithreaded

        len_inputs = 0
        
        if self.crop_type == CropType.standard:
            self.logger.debug("Cropping serialized")
            for slide_metadata in inputs:
                try:
                    self.tile_placeholders += self.section_manager.crop(slide_metadata, annotated_only=annotated_only)
                    len_inputs += 1
                except UtilException:
                    continue
        elif self.crop_type == CropType.pool_threading or self.crop_type == CropType.pool_multiprocessing:
            len_inputs, self.tile_placeholders =  self._pooling_cropping(inputs)

        print("*"*len("Found in total {} tiles.".format(len(self.tile_placeholders))))
        print("Found in total:\n {} tiles\n belonging to {} slides".format(len(self.tile_placeholders),
                                                                           len_inputs))
        print("*" * len("Found in total {} tiles.".format(len(self.tile_placeholders))))

    def _get_iter_list(self, inputs):

        for slide_metadata in inputs:
            yield (SlideManager(self.crop_size, overlap=self.overlap, verbose=self.verbose), slide_metadata, self.annotated_only)

    def _pooling_cropping(self, inputs):

        iter_list = self._get_iter_list(inputs)

        if self.crop_type == CropType.pool_threading:
            self.logger.debug("Cropping multithreaded")
            pool_fn_type = pathos.threading.ThreadPool
        else:
            self.logger.debug("Cropping multiprocessing")
            pool_fn_type = pathos.multiprocessing.ProcessPool

        with pool_fn_type() as pool:
            returns = pool.imap(pool_fn, iter_list, chunksize=5)

        returns = list(filter(lambda x: x is not None, returns))
        len_returns = len(returns)

        return len_returns, [section for section_list in returns for section in section_list]

    def _to_image(self, x):

        slide = openslide.OpenSlide(self.tile_placeholders[x].wsi_path)
        pil_object = slide.read_region([self.tile_placeholders[x].x,
                                        self.tile_placeholders[x].y],
                                       self.tile_placeholders[x].level,
                                       [self.tile_placeholders[x].size,
                                        self.tile_placeholders[x].size])
        pil_object = pil_object.convert('RGB')
        pil_object = pil_object.resize(size=(self.crop_size, self.crop_size))
        self.tile_placeholders[x].std = np.std(np.array(pil_object))
        label = self.tile_placeholders[x].label
        im_size = pil_object.size
        img = tf.reshape(tf.cast(pil_object.getdata(), dtype=tf.uint8), (im_size[0], im_size[1], 3))
        if self.tile_placeholders[x].std > self.information_threshold:
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
                                         name='label'),
                              tf.float32)

    def _fixup_shape(self, image: tf.Tensor, label: tf.Tensor):
        """
        Tensor.shape is determined at graph build time (tf.shape(tensor) gets you the runtime shape).
        In tf.numpy_function/tf.py_function “don't build a graph for this part, just run it in python”.
        So none of the code in such functions runs during graph building, and TensorFlow does not know the shape in there.
        With the function _fixup_shape we set the shape of the tensors.
        """
        image.set_shape([self.crop_size,
                         self.crop_size,
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
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def make_pytorch_dataset(self, remove_white: bool = True) -> Dataset:
        return WSIDatasetTorch(section_list=self.tile_placeholders, crop_size=self.crop_size, one_hot=self.one_hot, annotated_only=self.annotated_only)

    @property
    def get_tile_placeholders(self):
        return self.tile_placeholders

    @property
    def get_tile_placeholders_filt(self):
        return list(filter(lambda x: x.std > self.information_threshold, self.tile_placeholders))


def filt_tile_placeholders(tile_placeholders, threshold):
        return list(filter(lambda x: x.std > threshold, tile_placeholders))

def get_heatmap(tile_placeholders,
                slide : openslide.OpenSlide,
                class_to_map,
                num_classes,
                level_downsample,
                threshold : float = 0.5,
                tile_placeholders_mapping_key : str = 'prediction',
                colormap=cm.get_cmap('Blues')):

    """
    Builds a 3 channel map.
    The first three channels represent the sum of all the probabilities of the crops which contain that pixel
    belonging to classes 0-1, the fourth hold the number of crops which contain it.
    """

    _ , region_lv0 = get_region_lv0(slide)

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
            try:
                probabilities[top:top + side_y, left:left + side_x, 0:num_classes] = np.array(
                    tile[tile_placeholders_mapping_key][class_to_map])
            except KeyError:
                pass

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
    return slide_image, segmentation
