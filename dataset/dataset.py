import torch
from PIL import Image
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

transforms = A.Compose([
    A.Resize(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
    ToTensorV2(),
],
additional_targets={'image0':'image'})

class AnimeDataset(Dataset):
    def __init__(self, root_image, root_anime, transform=None):
        self.root_image = root_image
        self.root_anime = root_anime
        self.transform = transform

        self.anime_images = os.listdir(root_anime)
        self.image_images = os.listdir(root_image)
        self.length_dataset = max(len(self.anime_images), len(self.image_images)) # 1000, 1500
        self.amine_len = len(self.anime_images)
        self.image_len = len(self.image_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        anime_img = self.anime_images[index % self.amine_len]
        image_img = self.image_images[index % self.image_len]

        zebra_path = os.path.join(self.root_anime, anime_img)
        image_path = os.path.join(self.root_image, image_img)

        anime_img = np.array(Image.open(zebra_path).convert("RGB"))
        image_img = np.array(Image.open(image_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=anime_img, image0=image_img)
            anime_img = augmentations["image"]
            image_img = augmentations["image0"]

        return anime_img, image_img
