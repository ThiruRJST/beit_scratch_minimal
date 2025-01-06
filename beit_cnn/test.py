import glob

import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm

from train import ImageWoofClassifier
from albumentations.pytorch import ToTensorV2
import albumentations as A


transforms = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

def mask_random_patches(image, patch_mask_size, resized_image_size):
    resized_image = cv2.resize(image, (resized_image_size, resized_image_size))
    mask = np.ones((resized_image_size, resized_image_size,3), dtype=np.uint8)

    #generate random integer indices for the resized image size
    x_index, y_index = np.random.randint(0, resized_image_size-patch_mask_size, 2)
    mask[x_index:x_index+patch_mask_size, y_index:y_index+patch_mask_size] = 0

    labels = resized_image[x_index:x_index+patch_mask_size, y_index:y_index+patch_mask_size]

    patched_image = resized_image * mask

    return patched_image, labels

if __name__ == "__main__":
    model = ImageWoofClassifier.load_from_checkpoint("lightning_logs/version_3/checkpoints/epoch=9-step=1890.ckpt")
    model.eval()

    image_paths = glob.glob("../data/imagewoof2/val/*/*.JPEG")

    preds = []

    for path in tqdm(image_paths):
        image = cv2.imread(path)

        patched_image, labels = mask_random_patches(image, 24, 256)
        labels = labels / 255.0

        if transforms:
            patched_image = transforms(image=patched_image)['image']
        
        patch_tensor = patched_image.unsqueeze(0).to(model.device)
        
        labels_tensor = torch.from_numpy(labels)
        labels_tensor = labels_tensor.flatten().unsqueeze(0).to(model.device)

        yhat = model(patch_tensor)
    
        preds.append({"path":path, "mse":F.mse_loss(yhat, labels_tensor).item()})

    pd.DataFrame(preds).to_csv("masked_preds.csv", index=False)