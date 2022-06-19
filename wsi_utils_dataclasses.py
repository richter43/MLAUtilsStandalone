from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
from shapely.geometry import Polygon
from .ancillary_definitions import RenalCancerType
import os


@dataclass
class SlideMetadata:
    """Information about the slide (Location and such)
    """
    wsi_path: Optional[str] # Made optional after seeing that the ROI annotations do not depend on the original WSI
    annotation_path: str # Generalized for accepting WSI files aswell as XML files
    is_roi: bool # Boolean that facilitates the recognition of the annotation_path being either a WSI file or a XML file
    label: int #Label based on RenalCancerType enum

@dataclass
class PatientMetadata:
    wsi_root: str
    wsi_files: List[str]
    annotation_root: str
    ann_files: List[str]
    is_roi: bool
    diagnosis: str

    def get_slide_metadata(self):

      if self.is_roi:
        #Zipping is not used in this case due to the impossibility of ensuring a univocal map between roi files and wsi files 
        for ann_file in self.ann_files:
          #TODO: Change None to the real WSIs in the event further info is requried to be extracted from the WSIs
          yield SlideMetadata(None, os.path.join(self.annotation_root,ann_file), True, self.diagnosis)
      else:
        #Zipping works due to any wsi file being univocally mapped to a xml annotation
        for wsi_file, ann_file in zip(self.wsi_files, self.ann_files):
          yield SlideMetadata(os.path.join(self.wsi_root, wsi_file), os.path.join(self.annotation_root,ann_file), False, self.diagnosis)

    def __add__(self, other: PatientMetadata):
      assert self.wsi_root == other.wsi_root
      assert self.annotation_root == other.annotation_root
      assert self.is_roi == other.is_roi

      return PatientMetadata(self.wsi_root, self.wsi_files + other.wsi_files, self.annotation_root, self.ann_files + other.ann_files, self.is_roi, self.diagnosis)

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