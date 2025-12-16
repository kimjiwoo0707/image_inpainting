import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.utils.mask_generator import get_input_image

class CustomImageDataset(Dataset):
    def __init__(self, df, data_dir: str, mode: str, min_polygon_bbox_size: int = 50):
        self.df = df
        self.data_dir = data_dir
        self.mode = mode
        self.min_polygon_bbox_size = min_polygon_bbox_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        img_rel = self.df.iloc[idx, 0]
        img_path = os.path.join(self.data_dir, img_rel)

        if self.mode == "train":
            image = Image.open(img_path)
            return get_input_image(image, self.min_polygon_bbox_size)

        if self.mode == "valid":
            inp = np.load(img_path, allow_pickle=True).item()
            return inp

        if self.mode == "test":
            image = Image.open(img_path)
            return {"image_gray_masked": image}

        raise ValueError(f"Unknown mode: {self.mode}")
