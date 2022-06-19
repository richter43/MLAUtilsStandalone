import math
import os
from typing import List, Optional, Generator

import threading
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
import time

from .ancillary_definitions import RenalCancerType
from .annotation_utils_asap import get_points_xml_asap, get_region_lv0
from .annotation_utils_dataclasses import PointInfo
from .wsi_utils_dataclasses import Section, SlideMetadata


class WSIDatasetTorch(Dataset):

    def __init__(self, section_list: List[Section], crop_size: int, std_threshold: float, one_hot: bool = True, remove_white: bool = True, annotated_only: bool = False):
        super(WSIDatasetTorch, self).__init__()
        self.section_list : List[Section] = section_list
        self.crop_size = crop_size
        self.std_threshold = std_threshold
        self.num_classes = len(RenalCancerType)
        self.one_hot = one_hot
        self.max_threads = 32

        self.annotated_only = annotated_only
        if self.annotated_only == False:
            #we skip this computation for speedup as we don't need it (we assume the annotations don't depict the background of the wsi)
            self._parallel_compute_std()
        # self._compute_std_naive()
        
        self.remove_white = remove_white
        if remove_white and self.annotated_only == False:
            self._filter_white()
    
    def _compute_std_naive(self):

        init_time = time.perf_counter()

        previous_wsi_path = None

        for index in range(len(self.section_list)):
            section = self.section_list[index]
            
            if previous_wsi_path is not None and section.wsi_path == previous_wsi_path:
                pass
            else:
                slide = openslide.OpenSlide(section.wsi_path)

            previous_wsi_path = section.wsi_path
            pil_object = slide.read_region([section.x,
                                            section.y],
                                        section.level,
                                        [section.size,
                                            section.size])

            pil_object = pil_object.convert('RGB')

            section.std = np.std(np.array(pil_object))

        end_time = time.perf_counter()
        print(f"Time taken to compute std {end_time - init_time}")

    def _parallel_compute_std(self, step=32):
        """Computes standard deviation of the extracted image, this method is about 3x faster than the naive implementation

        Args:
            step (int, optional): _description_. Defaults to 32.
        """
        threads = []

        for idx in range(0,len(self.section_list),step):

            tmp_thread = threading.Thread(target=self._compute_std, args=[idx, step, len(self.section_list)])
            tmp_thread.start()
            threads.append(tmp_thread)

            if len(threads) == self.max_threads:
                threads.pop(0).join()

        for thread in threads:
            thread.join()

    def _compute_std(self, start, step, stop):

        previous_wsi_path = None

        for index in range(start, start+step if start + step < stop else stop):

            section = self.section_list[index]
            
            if previous_wsi_path is not None and section.wsi_path == previous_wsi_path:
                pass
            else:
                slide = openslide.OpenSlide(section.wsi_path)

            previous_wsi_path = section.wsi_path
            pil_object = slide.read_region([section.x,
                                            section.y],
                                        section.level,
                                        [section.size,
                                            section.size])

            pil_object = pil_object.convert('RGB')
            section.std = np.std(np.array(pil_object))


    def _filter_white(self):
        self.section_list = [section for section in self.section_list if section.std > self.std_threshold]

    def __len__(self):
        return len(self.section_list)

    def __getitem__(self, index):

        slide = openslide.OpenSlide(self.section_list[index].wsi_path)
        pil_object = slide.read_region([self.section_list[index].x,
                                        self.section_list[index].y],
                                       self.section_list[index].level,
                                       [self.section_list[index].size,
                                        self.section_list[index].size])
        pil_object = pil_object.convert('RGB')
        pil_object = pil_object.resize(size=(self.crop_size, self.crop_size))
        
        torch_tensor_convertor = transforms.ToTensor()
        img = torch_tensor_convertor(pil_object)
        if self.annotated_only == False:
            if self.section_list[index].std > self.std_threshold:
                label = torch.tensor(self.section_list[index].label)
            else:
                # Returning -1 means that the image looks like a white square
                label = torch.tensor(-1)
        else:
            #patch can never be white, label is always accurate (we also avoided computing the standard deviation for speedup)
            label = torch.tensor(self.section_list[index].label)
        if self.one_hot and label != -1:
            return img, F.one_hot(label, num_classes=self.num_classes)
        elif self.one_hot:
            return img, torch.zeros(self.num_classes)
        else: 
            return img, label

class SlideManager:
    def __init__(self, tile_size: int, overlap: bool = True, verbose: bool = False):
        """
        # SlideManager provides an easy way to generate a cropList object.
        # This object is not tied to a particular slide and can be reused to crop many slides using the same settings.
        """
        self.tile_size = tile_size
        self.level = 0
        self.overlap = int(1/overlap)
        self.verbose = verbose
        self.__sections: List[Section]= []

    # The usage of an encapsulating dunder doesn't seem to fit the use case here.
    def __generate_sections(self,
                             x_start: int,
                             y_start: int,
                             width: int,
                             height: int,
                             downsample_factor: float,
                             filepath: str):
        side = self.tile_size
        step = side // self.overlap
        # The usage of an encapsulating dunder doesn't seem to fit the use case here.
        self.__sections = []

        n_tiles = 0
        # N.B. Tiles are considered in the 0 level
        #TODO: There's no real need to pre-compute all of the boxes, an index should suffice to map from box to position 
        for y in range(0, height, step):
            for x in range(0, width, step):
                # x * step + side is right margin of the given tile
                if x + side > width or y + side > height:
                    continue
                n_tiles += 1
                self.__sections.append(Section(x=x_start + x, y= y_start + y, size=int(side // downsample_factor), level=self.level))
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

    def __crop_xml(self, slide_metadata: SlideMetadata, annotated_only: bool) -> List[Section]:

        slide = openslide.OpenSlide(slide_metadata.wsi_path)
        downsample = slide.level_downsamples[self.level]

        _ , (bounds_x, bounds_y, bounds_width, bounds_height) = get_region_lv0(slide)

        self.__generate_sections(bounds_x,
                                  bounds_y,
                                  bounds_width,
                                  bounds_height,
                                  downsample,
                                  slide_metadata.wsi_path)

        if annotated_only == True:
            point_info = PointInfo(get_points_xml_asap(slide_metadata.annotation_path))
        else:
            point_info = PointInfo(get_points_xml_asap(slide_metadata.annotation_path, "tumor"))

        patches_to_drop_i = []

        for i, index in enumerate(self.__sections):
            index.wsi_path = slide_metadata.wsi_path    

            # Assigns label depending on the intersection of the image with the annotation
            if not slide_metadata.is_roi:
                square = index.create_square_polygon()
                intercepting_polygons = point_info.strtree.query(square)

                label = -1
                for poly in intercepting_polygons:
                    if poly.intersection(square).area / square.area >= 0.5 and point_info.get_label_of_polygon(poly) == "tumor":
                        label = slide_metadata.label
                        break
                    elif poly.intersection(square).area / square.area >= 0.5 and annotated_only == True:
                        label = RenalCancerType.NOT_CANCER.value
                        break
                

            if annotated_only == False:
                #should be backward compatible
                index.label = label if label !=-1 else RenalCancerType.NOT_CANCER.value
            else:
                if label != -1:
                    #proceed normally
                    index.label = label
                else:
                    #mark the patch as dropped, it's not annotated and we don't want it
                    patches_to_drop_i.append(i)
        
        if annotated_only == True:
            #drop not annotated patches
            print(f"dropping {len(patches_to_drop_i)} non-annotated patches")
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
                                  slide_metadata.wsi_path)

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
            return self.__crop_roi(slide_metadata)
        else:
            return self.__crop_xml(slide_metadata, annotated_only)

        

class DatasetManager:
    def __init__(self,
                 inputs: Generator[SlideMetadata, SlideMetadata, None],
                 tile_size: int,
                 overlap: float = 1.0,
                 channels: int = 3,
                 batch_size: int = 32,
                 one_hot: bool = True,
                 std_threshold: int = 20,
                 annotated_only: bool = False,
                 verbose: bool = False):
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
        """

        self.crop_size = tile_size
        self.one_hot = one_hot
        self.overlap = overlap
        self.std_threshold = std_threshold
        self.num_classes = len(RenalCancerType)
        self.channels = channels
        self.batch_size = batch_size
        self.annotated_only = annotated_only
        self.section_manager = SlideManager(tile_size, overlap=self.overlap, verbose=verbose)
        self.tile_placeholders = [crop for slide_metadata in inputs for crop in self.section_manager.crop(slide_metadata, annotated_only= annotated_only)]

        print("*"*len("Found in total {} tiles.".format(len(self.tile_placeholders))))
        print("Found in total:\n {} tiles\n belonging to {} slides".format(len(self.tile_placeholders),
                                                                           len(inputs)))
        print("*" * len("Found in total {} tiles.".format(len(self.tile_placeholders))))

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
        if self.tile_placeholders[x].std > self.std_threshold:
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
        return WSIDatasetTorch(self.tile_placeholders, self.crop_size, self.std_threshold, self.one_hot, remove_white and self.annotated_only == False, annotated_only=self.annotated_only)

    @property
    def get_tile_placeholders(self):
        return self.tile_placeholders

    @property
    def get_tile_placeholders_filt(self):
        return list(filter(lambda x: x.std > self.std_threshold, self.tile_placeholders))


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
    # segmentation[segmentation != 0] = 255
    return slide_image, segmentation