from dataclasses import dataclass, field
from typing import Optional
from shapely.geometry import Polygon

@dataclass
class SlideMetadata:
    """asd
    """
    wsi_path: str
    xml_path: Optional[str] = None
    label: Optional[str] = None

@dataclass
class Section:
    top_coord: int
    left_coord: int
    size: int
    level: int
    wsi_path: str = field(init=False)
    label: str = field(init=False)
    
    def create_square_polygon(self) -> Polygon:
    
        points = [(self.top_coord, self.left_coord), (self.top_coord, self.left_coord  + self.size), (self.top_coord + self.size, self.left_coord + self.size), (self.top_coord + self.size, self.left_coord)]

        return Polygon(points)