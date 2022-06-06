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
    x: int
    y: int
    size: int
    level: int
    wsi_path: str = field(init=False)
    label: str = field(init=False)
    std: float = field(init=False)
    
    def create_square_polygon(self) -> Polygon:
    
        points = [(self.x, self.y), (self.x, self.y  + self.size), (self.x + self.size, self.y + self.size), (self.x + self.size, self.y)]

        return Polygon(points)