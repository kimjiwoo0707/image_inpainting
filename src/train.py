import os
import zipfile
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import lightning as L

from glob import glob
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image

from src.models.unet_pp_cbam import CBAMUnetPlusPlus
from src.datasets.dataset import CustomImageDataset
from src.datasets.collate import CollateFn
from src.metrics.metrics import get_ssim_score, get_masked_ssim_score, get_histogram_similarity
from src.utils.mask_generator import get_input_image

class LitIRModel(L.LightningModule):
    def __init__(self, model_1, model_2, image_mean=0.5, image_std=0.225, max_epochs=50, lr=1e-4):
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.image_mean = image_mean
        self.image_std = image_std
        self.max_epochs = max_epochs
        self.lr = lr

    def forward(self, images_gray_masked):
        images_gray_restored = self.model_1(images_gray_masked) + images_gray_masked
        images_restored = self.model_2(images_gray_restored)
        return images_gray_restored, images_restored

    def unnormalize(self, x, round=False):
        x = ((x * self.image_std + self.image_mean) * 255).clamp(0, 255)
        if round:
            x = torch.round(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        num_training_steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=3,
            num_training_steps=num_training_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def training_step(self, batch, batch_idx):
        images_gray_masked = batch["images_gray_masked"]
        images_gray = batch["images_gray"]
        images_gt = batch["images_gt"]

        images_gray_restored, images_restored = self(images_gray_masked)

        loss_gray = 0.5 * F.l1_loss(images_gray, images_gray_restored) + 0.5 * F.mse_loss(images_gray, images_gray_restored)
        loss_rgb = 0.5 * F.l1_loss(images_gt, images_restored) + 0.5 * F.mse_loss(images_gt, images_restored)
        loss = 0.5 * loss_gray + 0.5 * loss_rgb

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        masks = batch["masks"]
        images_gray_masked = batch["images_gray_masked"]
        images_gt = batch["images_gt"]

        _, images_restored = self(images_gray_masked)

        images_gt_u8 = self.unnormalize(images_gt, round=True)
        images_restored_u8 = self.unnormalize(images_restored, round=True)

        masks_np = masks.detach().cpu().numpy()
        gt_np = images_gt_u8.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
        pr_np = images_restored_u8.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)

        total_ssim = 0.0
        masked_ssim = 0.0
        hist_sim = 0.0

        bs = len(gt_np)
        for g, p, m in zip(gt_np, pr_np, masks_np):
            total_ssim += get_ssim_score(g, p) / bs
            masked_ssim += get_masked_ssim_score(g, p, m) / bs
            hist_sim += get_histogram_similarity(g, p) / bs

        score = total_ssim * 0.2 + masked_ssim * 0.4 + hist_sim * 0.4
        self.log("val_score", score, prog_bar=True)
        self.log("val_total_ssim", total_ssim)
        self.log("val_masked_ssim", masked_ssim)
        self.log("val_hist_sim", hist_sim)
        return score


def build_valid_cache(train_df, train_data_dir, valid_cache_dir, seed=42, min_polygon_bbox_size=50):
    os.makedirs(valid_cache_dir, exist_ok=True)
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Build VALID cache"):
        img_rel = row.iloc[0]
        img_path = os.path.join(train_data_dir, img_rel)

        save_name = os.path.basename(img_path).replace("TRAIN", "VALID").replace("png", "npy")
        save_path = os.path.join(valid_cache_dir, save_name)
        if os.path.exists(save_path):
            continue

        image = Image.open(img_path)
        sample = get_input_image(image, min_polygon_bbox_size=min_polygon_bbox_size)
        np.save(save_path, sample)


def main():
    SEED = 42
    N_SPLIT = 5
    BATCH_SIZE = 8
    IMAGE_MEAN = 0.5
    IMAGE_STD = 0.225
    MIN_POLYGON_BBOX_SIZE = 50

    TRAIN_DATA_DIR = "/home/work/jiu/open/train_gt"
    TEST_DATA_DIR = "/home/work/jiu/open/test_input"
    VALID_CACHE_DIR = f"/home/work/jiu/open/valid_input/SEED{SEED}_MIN{MIN_POLYGON_BBOX_SIZE}"

    train_df = pd.read_csv("/home/work/jiu/preproc/train_preproc.csv")
    test_df = pd.read_csv("/home/work/jiu/preproc/test_preproc.csv")

    L.seed_everything(SEED)

    build_valid_cache(train_df, TRAIN_DATA_DIR, VALID_CACHE_DIR, seed=SEED, min_polygon_bbox_size=MIN_POLYGON_BBOX_SIZE)

    train_df = train_df[train_df["label"] != -1].reset_index(drop=True)

    kf = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)
    fold_idx = 0
    train_idx, valid_idx = list(kf.split(train_df["image"], train_df["label"]))[fold_idx]

    train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
    valid_fold_df = train_df.iloc[valid_idx].reset_index(drop=True)
    valid_fold_df["image"] = valid_fold_df["image"].apply(lambda x: x.replace("TRAIN", "VALID").replace("png", "npy"))

    train_ds = CustomImageDataset(train_fold_df, data_dir=TRAIN_DATA_DIR, mode="train", min_polygon_bbox_size=MIN_POLYGON_BBOX_SIZE)
    valid_ds = CustomImageDataset(valid_fold_df, data_dir=VALID_CACHE_DIR, mode="valid", min_polygon_bbox_size=MIN_POLYGON_BBOX_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=CollateFn(IMAGE_MEAN, IMAGE_STD, "train"))
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False, collate_fn=CollateFn(IMAGE_MEAN, IMAGE_STD, "valid"))

    model_1 = CBAMUnetPlusPlus("efficientnet-b4", "imagenet", in_channels=1, classes=1)
    model_2 = CBAMUnetPlusPlus("efficientnet-b4", "imagenet", in_channels=1, classes=3)
    lit_model = LitIRModel(model_1=model_1, model_2=model_2, image_mean=IMAGE_MEAN, image_std=IMAGE_STD, max_epochs=50, lr=1e-4)

    ckpt_cb = ModelCheckpoint(
        monitor="val_score",
        mode="max",
        dirpath="./checkpoints",
        filename=f"cbam-unetpp-fold{fold_idx}-seed{SEED}" + "-{epoch:02d}-{val_score:.4f}",
        save_top_k=1,
        save_weights_only=True,
    )
    es_cb = EarlyStopping(monitor="val_score", mode="max", patience=7)

    trainer = L.Trainer(
        max_epochs=50,
        precision="bf16-mixed",
        callbacks=[ckpt_cb, es_cb],
        detect_anomaly=False,
    )

    trainer.fit(lit_model, train_loader, valid_loader)

if __name__ == "__main__":
    main()
