from dataclasses import dataclass,field
from typing import List, Tuple
from shapely.strtree import STRtree
from shapely.geometry import Polygon

@dataclass
class AnnotationData:
    name: str
    group_name: str
    points: List[Tuple[int,int]]
    polygon: Polygon = field(init=False)
    def __post_init__(self):
        self.polygon = Polygon(self.points)

@dataclass
class PointInfo:
    ann_list: List[AnnotationData]
    strtree: STRtree = field(init=False)

    def __post_init__(self):
        polygons = [ann.polygon for ann in self.ann_list]
        self.strtree = STRtree(polygons)

    def get_label_of_polygon(self, polygon: Polygon):
        
        for ann in self.ann_list:
            if ann.polygon is polygon:
                return ann.group_name 