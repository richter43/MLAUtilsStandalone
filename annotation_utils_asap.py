import sys
from typing import Dict, List, Tuple, Union, Optional
import os

import cv2
import matplotlib.pyplot as plt
# Make sure to add mir's location to PYTHONPATH (ASAP's directory)
import json
import numpy as np
from matplotlib import cm
from openslide import PROPERTY_NAME_BACKGROUND_COLOR, OpenSlide
from PIL import Image
from shapely.geometry import Polygon

from .annotation_utils_dataclasses import AnnotationData, PointInfo

module_dir = os.path.dirname(__file__)
with open(os.path.join(module_dir, "settings.json")) as f:
    open_json = json.load(f)
    sys.path.append(open_json["ASAP_PATH"])

import multiresolutionimageinterface as mir

def get_region_lv0(slide: OpenSlide) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:

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
        
        coords = (bounds_x, bounds_y)

    else:
        # If bounding box of the non-empty region of the slide is not available
        # Slide dimensions of given level reported to level 0
        coords = (0,0)
        region_lv0 = (0, 0, slide.level_dimensions[0][0], slide.level_dimensions[0][1])

    return coords, region_lv0

def get_points_base_asap(xml_file: str) -> PointInfo:
    
    ann_list = get_points_xml_asap(xml_file)

    return PointInfo(ann_list)

def get_annotationdata_list(annotation_list: mir.AnnotationList, selected_group: Optional[str]) -> List[AnnotationData]:

    groups = [group.getName() for group in annotation_list.getGroups()]

    if selected_group is not None:
        assert selected_group in groups

    ann_list: List[AnnotationData] = []

    for annotation in annotation_list.getAnnotations():

        group_name = annotation.getGroup().getName()

        if group_name is not None and selected_group is not None and group_name != selected_group:
            continue

        points_annotation = []
        for coordinate in annotation.getCoordinates():
            points_annotation.append((coordinate.getX(), coordinate.getY()))

        name = annotation.getName()
        new_ann = AnnotationData(name=name, group_name=group_name, points=points_annotation)    
        ann_list.append(new_ann)

    return ann_list

def get_points_xml_asap(xml_file: str, selected_group: Optional[str] = None) -> List[AnnotationData]:

    #Memory space in which annotations will be kept at
    annotation_list = mir.AnnotationList()

    #ASAP's XML manager
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(xml_file)
    xml_repository.load()

    return get_annotationdata_list(annotation_list, selected_group)

def get_annotation_mask_asap(xml_file: str, slide: OpenSlide, level_downsample: float) -> np.array:

    """Returns the mask of a tile"""
    
    coords, region_lv0 = get_region_lv0(slide)
    region_lv0 = [round(x) for x in region_lv0]
    region_lv_selected = [round(x * level_downsample) for x in region_lv0]

    mask = np.zeros((region_lv_selected[3], region_lv_selected[2]), dtype=np.uint8)

    point_info = get_points_base_asap(xml_file)

    points = [[(int(p[0] * level_downsample), int(p[1] * level_downsample)) for p in annotation.points] for annotation in
                point_info.ann_list]
    coords = tuple([int(c * level_downsample) for c in coords])
    points = [[(int(p[0] - coords[0]), int(p[1] - coords[1])) for p in pointSet] for pointSet in points]
    for pointSet in points:
        cv2.fillPoly(mask, [np.array(pointSet, dtype=np.int32)], 255)

    return mask

def get_thumbnail_offset(slide: OpenSlide, size: Tuple[int, int]):

    #Get actual location of the image in the WSI
    bounds_width = int(slide.properties['openslide.bounds-width'])
    bounds_height = int(slide.properties['openslide.bounds-height'])
    coord_lv0 = (int(slide.properties['openslide.bounds-x']),int(slide.properties['openslide.bounds-y']))
    #Get best downscaling factor for the requested image size
    downsample = max(*[dim / thumb for dim, thumb in zip(slide.dimensions, size)])
    #Get the level for which the downscaling corresponds to
    level = slide.get_best_level_for_downsample(downsample)
    #Compute the corresponding image size in the requested zoom level
    image_size_at_desired_level = (int(bounds_width/slide.level_downsamples[level]), int(bounds_height/slide.level_downsamples[level]))
    #Read the image
    tile = slide.read_region(coord_lv0, level, image_size_at_desired_level)

    # Apply on solid background
    bg_color = '#' + slide.properties.get(PROPERTY_NAME_BACKGROUND_COLOR,
            'ffffff')
    thumb = Image.new('RGB', tile.size, bg_color)
    thumb.paste(tile, None, tile)
    thumb.thumbnail(size, Image.ANTIALIAS)
    return thumb

def overlap_asap(slide: OpenSlide, mask: np.array, level_downsample: float, colormap=cm.get_cmap('Blues')) -> Image:

    coords, region_lv0 = get_region_lv0(slide)
    region_lv0 = [round(x) for x in region_lv0]
    region_lv_selected = [round(x * level_downsample) for x in region_lv0]

    downscaled_size = (region_lv_selected[2], region_lv_selected[3])

    if 'openslide.bounds-width' in slide.properties.keys():
        slide_image = get_thumbnail_offset(slide, downscaled_size)
    else:
        slide_image = slide.get_thumbnail(downscaled_size)
    
    map_ = colormap(mask)
    #TODO: I feel like the resizing of the image is too crude
    roi_map = Image.fromarray((map_ * 255).astype('uint8')).resize(slide_image.size)
    roi_map.putalpha(75)
    slide_image = slide_image.convert('RGBA')
    slide_image.alpha_composite(roi_map)

    return slide_image