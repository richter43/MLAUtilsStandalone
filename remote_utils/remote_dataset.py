import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.datasets import DatasetFolder
import torchvision.transforms as transforms

from typing import Optional, Callable, Tuple, Any, Dict

from .download_utils import download_decrypt_untar
from ..wsi_utils_dataclasses import PatientMetadata

def byte_to_default(x):
    default_float_dtype = torch.get_default_dtype()
    return x.to(dtype=default_float_dtype).div(255)

class PatientImagesDataset(DatasetFolder):

    def __init__(self,
        patient_id: str,
        root: str,
        class_to_idx: Dict[str, int],
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        imagenet_pretrain: bool = False
        ) -> None:

        extensions = [".jpg"]
        
        # Required to map from ByteTensor to Tensor with Floats (No idea why read_image doesn't already do so by default)
        transform_list = [byte_to_default]

        if imagenet_pretrain:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            transform_list.append(normalize)

        transform = transforms.Compose(transform_list)

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

