import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, data_dir, mode='train', transform=None):
        """
        Custom dataset for loading images and preprocessing.

        Args:
            dataframe (pd.DataFrame): Contains image file paths and labels.
            data_dir (str): Directory containing the images.
            mode (str): Mode of dataset - 'train', 'valid', 'test'.
            transform (callable, optional): Transformations to apply to images.
        """
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Fetches an item by index.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: Contains 'image', 'label', and optional 'mask' for training or validation.
        """
        # Load the image
        img_path = os.path.join(self.data_dir, self.dataframe.iloc[idx]['image'])
        image = np.array(Image.open(img_path).convert('L'))  # Convert to grayscale (1 channel)

        # Normalize the image to [0, 1]
        image = image / 255.0

        # Load the label if mode is 'train' or 'valid'
        if self.mode in ['train', 'valid']:
            label_path = os.path.join(self.data_dir, self.dataframe.iloc[idx]['label'])
            label = np.array(Image.open(label_path).convert('L')) / 255.0

            # If transform is provided, apply it
            if self.transform:
                image = self.transform(image)
                label = self.transform(label)

            return {'image': torch.tensor(image, dtype=torch.float32).unsqueeze(0),
                    'label': torch.tensor(label, dtype=torch.float32).unsqueeze(0)}
        
        elif self.mode == 'test':
            # If transform is provided, apply it
            if self.transform:
                image = self.transform(image)

            return {'image': torch.tensor(image, dtype=torch.float32).unsqueeze(0)}

        else:
            raise ValueError("Mode must be 'train', 'valid', or 'test'")

# Optional Collate Function
class CollateFn:
    def __init__(self, mode='train'):
        """
        Collate function for DataLoader.

        Args:
            mode (str): Mode of dataset - 'train', 'valid', 'test'.
        """
        self.mode = mode

    def __call__(self, batch):
        """
        Collates a batch of data.

        Args:
            batch (list): List of items fetched by the dataset.

        Returns:
            dict: Batched data with images and labels/masks if available.
        """
        images = torch.stack([item['image'] for item in batch])
        
        if self.mode in ['train', 'valid']:
            labels = torch.stack([item['label'] for item in batch])
            return {'images': images, 'labels': labels}
        elif self.mode == 'test':
            return {'images': images}
        else:
            raise ValueError("Mode must be 'train', 'valid', or 'test'")
