import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, data_dir, mode='train', transform=None):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.dataframe.iloc[idx]['image'])
        image = np.array(Image.open(img_path).convert('L'))  # Convert to grayscale (1 channel)
        image = image / 255.0
        
        if self.mode in ['train', 'valid']:
            label_path = os.path.join(self.data_dir, self.dataframe.iloc[idx]['label'])
            label = np.array(Image.open(label_path).convert('L')) / 255.0

            if self.transform:
                image = self.transform(image)
                label = self.transform(label)

            return {'image': torch.tensor(image, dtype=torch.float32).unsqueeze(0),
                    'label': torch.tensor(label, dtype=torch.float32).unsqueeze(0)}
        
        elif self.mode == 'test':
            if self.transform:
                image = self.transform(image)

            return {'image': torch.tensor(image, dtype=torch.float32).unsqueeze(0)}

        else:
            raise ValueError("Mode must be 'train', 'valid', or 'test'")

class CollateFn:
    def __init__(self, mode='train'):
        self.mode = mode

    def __call__(self, batch):
        images = torch.stack([item['image'] for item in batch])
        
        if self.mode in ['train', 'valid']:
            labels = torch.stack([item['label'] for item in batch])
            return {'images': images, 'labels': labels}
        elif self.mode == 'test':
            return {'images': images}
        else:
            raise ValueError("Mode must be 'train', 'valid', or 'test'")
