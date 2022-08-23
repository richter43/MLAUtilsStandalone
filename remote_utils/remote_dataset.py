from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torchvision.io import read_image
import numpy as np
import albumentations as A

from ..wsi_utils_dataclasses import PatientMetadata
from .download_utils import download_decrypt_untar

def transform_callable(image):
  
  transformed_image = transform(image=image)['image']
  return transformed_image

def byte_to_default(x):
    # Required to map from ByteTensor to Tensor with Floats (No idea why read_image doesn't already do so by default)
    default_float_dtype = torch.get_default_dtype()
    return x.to(dtype=default_float_dtype).div(255)

class PatientImagesDataset(DatasetFolder):

    def __init__(self,
        patient_id: str,
        root: str,
        class_to_idx: Dict[str, int],
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        imagenet_pretrain: bool = False,
        augment: bool = False
        ) -> None:

        extensions = [".jpg"]
        
        if augment:
            #I'd prefer the usage of a lambda function, however, PEPs disallow me to do so :(
            def read_image_callback(image_path: str) -> np.ndarray:
                image = read_image(image_path).numpy().transpose(1,2,0)
                return image
            self.loader = read_image_callback
            transform = TransformImage(imagenet_pretrain=imagenet_pretrain)
        else:
            self.loader = read_image
            transform_list = [byte_to_default]
            if imagenet_pretrain:
                transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            transform = transforms.Compose(transform_list)
                

        super(DatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        
        classes, _ = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.patient_id = patient_id

        
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

class TransformImage:

    def __init__(self, imagenet_pretrain: bool):
        super(TransformImage, self).__init__()

        if imagenet_pretrain:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            mean = (0, 0, 0)
            std = (1, 1, 1)

        self.transform = A.Compose([
                        A.RandomRotate90(),
                        A.CLAHE(),
                        A.ElasticTransform(alpha=120, sigma=120 * 0.1, alpha_affine=120 * 0.03),
                        A.RandomBrightnessContrast(),
                        A.Blur(blur_limit=3),
                        A.GaussNoise(),
                        A.Normalize(mean=mean, std=std),
                        A.pytorch.ToTensorV2()
                        ])
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:

        transformed_image = self.transform(image=image)['image']

        return transformed_image