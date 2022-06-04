from dataclasses import dataclass,field
from typing import List, Tuple
from shapely.strtree import STRtree
from shapely.geometry import Polygon

@dataclass(frozen=True)
class AnnotationData:
    name: str
    points: List[Tuple[int,int]]

@dataclass
class PointInfo:
    ann_list: List[AnnotationData]
    polygons: List[Polygon] = field(init=False)
    strtree: STRtree = field(init=False)

    def __post_init__(self):
        self.polygons = [Polygon(ann.points) for ann in self.ann_list]
        self.strtree = STRtree(self.polygons)