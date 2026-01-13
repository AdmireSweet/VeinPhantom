# VeinDataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import random


def deepen_vein(img_u8: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cla = clahe.apply(img_u8)
    bg = cv2.GaussianBlur(cla, (0, 0), 15)
    enh = cv2.addWeighted(cla, 1.7, bg, -0.7, 0)
    gamma = 0.85
    table = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    enh = cv2.LUT(enh, table)
    enh = cv2.normalize(enh, None, 0, 255, cv2.NORM_MINMAX)
    return enh.astype(np.uint8)


class PalmVeinDataset(Dataset):
    IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

        class_names = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        self.samples = []
        for class_name in class_names:
            class_dir = os.path.join(root, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(self.IMG_EXTS):
                    self.samples.append(
                        (os.path.join(class_dir, fname), self.class_to_idx[class_name])
                    )

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_u8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if random.random() < 0.9:
            img_u8 = deepen_vein(img_u8)

        ten = transforms.ToTensor()(img_u8)
        ten = ten.repeat(3, 1, 1)
        return ten, label
