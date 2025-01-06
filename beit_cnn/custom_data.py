import albumentations as A
import cv2
import numpy as np
import torch

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class ImageWoofDataset(Dataset):
    def __init__(self, data, patch_size:int = 24):
        self.data = data
        self.patch_size = patch_size
        
        self.transforms = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path = f"../data/imagewoof2/{self.data[idx]}"
        
        image = cv2.imread(path)
        patched_image, labels = self.mask_random_patches(image, self.patch_size, 256)
        labels = labels / 255.0
        if self.transforms:
            image = self.transforms(image=patched_image)['image']
        
        return image, torch.tensor(labels, dtype=torch.float32).flatten()
    
    def mask_random_patches(self, image, patch_mask_size, resized_image_size):
        resized_image = cv2.resize(image, (resized_image_size, resized_image_size))
        mask = np.ones((resized_image_size, resized_image_size,3), dtype=np.uint8)

        #generate random integer indices for the resized image size
        x_index, y_index = np.random.randint(0, resized_image_size-patch_mask_size, 2)
        mask[x_index:x_index+patch_mask_size, y_index:y_index+patch_mask_size] = 0

        labels = resized_image[x_index:x_index+patch_mask_size, y_index:y_index+patch_mask_size]

        patched_image = resized_image * mask

        return patched_image, labels