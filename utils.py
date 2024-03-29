import os

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pathos
import seaborn as sns
from matplotlib import cm as plt_cmap
from PIL import Image
from shapely.geometry import Polygon
import enum
from functools import wraps
import time

from .ancillary_definitions import RenalCancerType
from .wsi_utils_dataclasses import Section


def seaborn_cm(cm, ax, tick_labels, fontsize=14):

    group_counts = ["{:0.0f}".format(value) for value in cm.flatten()]
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.nan_to_num(cm)
    group_percentages = ["{:0.0f}".format(value*100) for value in cm.flatten()]
    cm_labels = [f"{c}\n{p}%" for c, p in zip(group_counts, group_percentages)]
    cm_labels = np.asarray(cm_labels).reshape(len(tick_labels), len(tick_labels))
    sns.heatmap(cm,
                ax=ax,
                annot=cm_labels,
                fmt='',
                cbar=False,
                cmap=plt_cmap.Greys,
                linewidths=1, linecolor='black',
                annot_kws={"fontsize": fontsize},
                xticklabels=tick_labels,
                yticklabels=tick_labels)
    ax.set_yticklabels(ax.get_yticklabels(), size=fontsize, rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), size=fontsize, rotation=45)

def get_label_from_path(path: str):

    norm_path = path.lower()

    if 'onco' in norm_path:
        return RenalCancerType.ONCOCYTOMA.value
    elif 'chromo' in norm_path:
        return RenalCancerType.CHROMOPHOBE.value
    elif 'prcc' in norm_path:
        return RenalCancerType.PAPILLARY.value
    else:
        return RenalCancerType.CLEAR_CELL.value

def image_entropy(slide: openslide.OpenSlide, section: Section):

    pil_obj = slide_read_region(slide, section.x, section.y, section.level, section.size)
    entropy = pil_obj.entropy() ** 2
    return entropy

def slide_read_region(slide: openslide.OpenSlide, x: int, y: int, level: int, size: int) -> Image:

    pil_object = slide.read_region([x, y], level, [size, size])
    pil_object = pil_object.convert('RGB')

    return pil_object

class UtilException(BaseException):
    pass

class SelectedGroupNotFound(UtilException):
    pass

class InitializationException(UtilException):
    pass

class TarException(UtilException):
    pass

class CropType(enum.Enum):
    standard = 0
    pool_threading = 1
    pool_multiprocessing = 2


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper