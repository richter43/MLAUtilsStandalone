from torch.utils.data import Dataset, ConcatDataset
from torchvision.io import read_image
from torchvision.datasets import DatasetFolder

from typing import Optional, Callable, Tuple, Any, Dict

from .download_utils import download_decrypt_untar
from ..ancillary_definitions import RenalCancerType
from ..wsi_utils_dataclasses import PatientMetadata

class PatientImagesDataset(DatasetFolder):

    def __init__(self,
        patient_id: str,
        root: str,
        class_to_idx: Dict[str, int],
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        ) -> None:

        extensions = [".jpg"]

        transform = None

        super(DatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        classes, _ = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.patient_id = patient_id

        self.loader = read_image
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

