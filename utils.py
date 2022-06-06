import numpy as np
import seaborn as sns
from matplotlib import cm as plt_cmap
from shapely.geometry import Polygon
from .ancillary_definitions import RenalCancerType

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
