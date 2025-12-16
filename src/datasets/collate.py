import numpy as np
import torch

class CollateFn:
    def __init__(self, mean: float = 0.5, std: float = 0.225, mode: str = "train"):
        self.mean = mean
        self.std = std
        self.mode = mode

    def __call__(self, examples):
        if self.mode in ("train", "valid"):
            masks, images_gray, images_gray_masked, images_gt = [], [], [], []

            for ex in examples:
                masks.append(ex["mask"])
                images_gray.append(self.normalize_image(ex["image_gray"]))
                images_gray_masked.append(self.normalize_image(ex["image_gray_masked"]))
                images_gt.append(self.normalize_image(np.array(ex["image_gt"])))

            return {
                "masks": torch.from_numpy(np.stack(masks)).long(),
                "images_gray": torch.from_numpy(np.stack(images_gray)).unsqueeze(1).float(),
                "images_gray_masked": torch.from_numpy(np.stack(images_gray_masked)).unsqueeze(1).float(),
                "images_gt": torch.from_numpy(np.stack(images_gt)).permute(0, 3, 1, 2).float(),
            }

        if self.mode == "test":
            imgs = [self.normalize_image(ex["image_gray_masked"]) for ex in examples]
            return {"images_gray_masked": torch.from_numpy(np.stack(imgs)).unsqueeze(1).float()}

        raise ValueError(f"Unknown mode: {self.mode}")

    def normalize_image(self, image):
        return (np.array(image) / 255.0 - self.mean) / self.std
