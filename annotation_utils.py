import cv2
import numpy as np
from matplotlib import cm
import xml.etree.ElementTree as ET
from shapely.strtree import STRtree
from shapely.geometry import Polygon
from PIL import Image


def get_points_base(xml_file, colors_to_use, custom_colors=[]):
    color_key = ''.join([k for k in colors_to_use])
    points, map_idx = get_points_xml(xml_file,
                                     colors_to_use,
                                     custom_colors)
    point_polys = [Polygon(point) for point in points]
    return {'points': points.copy(),
            'map_idx': map_idx.copy(),
            'polygons': point_polys.copy(),
            'STRtree': STRtree(point_polys)}


def color_ref_match_xml(colors_to_use, custom_colors):
    """Given a string or list of strings corresponding to colors to use, returns the hexcodes of those colors"""

    color_ref = [(65535, 1, 'yellow'), (65280, 2, 'green'), (255, 3, 'red'), (16711680, 4, 'blue'),
                 (16711808, 5, 'purple'), (np.nan, 6, 'other')] + custom_colors

    if colors_to_use is not None:

        if isinstance(colors_to_use, str):
            colors_to_use = colors_to_use.lower()
        else:
            colors_to_use = [color.lower() for color in colors_to_use]

        color_map = [c for c in color_ref if c[2] in colors_to_use]
    else:
        color_map = color_ref

    return color_map


def get_points_xml(xml_file, colors_to_use, custom_colors):
    """Given a set of annotation colors, parses the xml file to get those annotations as lists of verticies"""
    color_map = color_ref_match_xml(colors_to_use, custom_colors)

    color_key = ''.join([k[2] for k in color_map])
    full_map = color_ref_match_xml(None, custom_colors)

    # create element tree object
    tree = ET.parse(xml_file)

    # get root element
    root = tree.getroot()

    map_idx = []
    points = []

    for annotation in root.findall('Annotation'):
        line_color = int(annotation.get('LineColor'))
        mapped_idx = [item[1] for item in color_map if item[0] == line_color]

        if not mapped_idx and not [item[1] for item in full_map if item[0] == line_color]:
            if 'other' in [item[2] for item in color_map]:
                mapped_idx = [item[1] for item in color_map if item[2] == 'other']

        if mapped_idx:
            if isinstance(mapped_idx, list):
                mapped_idx = mapped_idx[0]

            for regions in annotation.findall('Regions'):
                for annCount, region in enumerate(regions.findall('Region')):
                    map_idx.append(mapped_idx)

                    for vertices in region.findall('Vertices'):
                        points.append([None] * len(vertices.findall('Vertex')))
                        for k, vertex in enumerate(vertices.findall('Vertex')):
                            points[-1][k] = (int(float(vertex.get('X'))), int(float(vertex.get('Y'))))

    sort_order = [x[1] for x in color_map]
    new_order = []
    for x in sort_order:
        new_order.extend([index for index, v in enumerate(map_idx) if v == x])

    points = [points[x] for x in new_order]
    map_idx = [map_idx[x] for x in new_order]

    return points, map_idx


def get_annotation_mask(xml_file,
                        slide,
                        level_downsample,
                        colors_to_use=None):
    """Returns the mask of a tile"""
    coords = (0, 0)
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

    points_dict = get_points_base(xml_file, colors_to_use)
    points = points_dict['points']
    map_idx = points_dict['map_idx']
    point_polys = points_dict['polygons']
    point_tree = points_dict['STRtree']

    tile_poly = Polygon([(coords[0], coords[1]),
                         (coords[0], coords[1] + region_lv0[3]),
                         (coords[0] + region_lv0[2], coords[1] + region_lv0[3]),
                         (coords[0] + region_lv0[2], coords[1])])
    mask = np.zeros((region_lv_selected[3], region_lv_selected[2]), dtype=np.uint8)

    if not point_polys:
        point_polys = [Polygon(point) for point in points]
        point_tree = STRtree(point_polys)

    index_by_id = dict((id(pt), i) for i, pt in enumerate(point_polys))
    intersecting_points = [index_by_id[id(pt)] for pt in point_tree.query(tile_poly)]

    points_maps = [point_map for idx, point_map in enumerate(zip(points, map_idx)) if idx in intersecting_points]
    if points_maps:
        points, map_idx = zip(*points_maps)
        points = [[(int(p[0] * level_downsample), int(p[1] * level_downsample)) for p in pointSet] for pointSet in
                  points]
        coords = tuple([int(c * level_downsample) for c in coords])
        points = [[(int(p[0] - coords[0]), int(p[1] - coords[1])) for p in pointSet] for pointSet in points]
        for annCount, pointSet in enumerate(points):
            cv2.fillPoly(mask, [np.asarray(pointSet).reshape((-1, 1, 2))], map_idx[annCount])
    mask[mask!=0] = 255
    return mask


def overlap(slide, mask, level_downsample,colormap=cm.get_cmap('Blues')):

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
    slide_image = slide.get_thumbnail((region_lv_selected[2], region_lv_selected[3]))
    map_ = colormap(mask)
    roi_map = Image.fromarray((map_ * 255).astype('uint8'))
    roi_map.putalpha(75)
    slide_image = slide_image.convert('RGBA')
    slide_image.alpha_composite(roi_map)
    slide_image.convert('RGBA')
    return slide_image
