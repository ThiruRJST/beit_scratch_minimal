import albumentations as A
import cv2

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class DVAEDataset(Dataset):

    def __init__(self, image_paths):
        super().__init__()

        self.image_paths = image_paths
        self.transforms = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path_prefix = "../data/imagewoof2/"
        image_path = path_prefix + self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        return image
        
