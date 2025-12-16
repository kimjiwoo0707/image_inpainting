import os
import zipfile
import numpy as np
import pandas as pd
import lightning as L
from glob import glob
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from src.models.unet_pp_cbam import CBAMUnetPlusPlus
from src.train import LitIRModel
from src.datasets.dataset import CustomImageDataset
from src.datasets.collate import CollateFn

def main():
    IMAGE_MEAN = 0.5
    IMAGE_STD = 0.225
    BATCH_SIZE = 16

    TEST_DATA_DIR = "/home/work/jiu/open/test_input"
    SUBMISSION_DIR = "./submission/CBAMUnetPlusPlus"
    SUBMISSION_ZIP = "./submission/CBAMUnetPlusPlus.zip"

    test_df = pd.read_csv("/home/work/jiu/preproc/test_preproc.csv")

    test_ds = CustomImageDataset(test_df, data_dir=TEST_DATA_DIR, mode="test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=CollateFn(IMAGE_MEAN, IMAGE_STD, "test"))

    model_1 = CBAMUnetPlusPlus("efficientnet-b4", "imagenet", in_channels=1, classes=1)
    model_2 = CBAMUnetPlusPlus("efficientnet-b4", "imagenet", in_channels=1, classes=3)

    ckpts = sorted(glob("./checkpoints/*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoint found in ./checkpoints/")
    ckpt_path = ckpts[-1]

    lit_model = LitIRModel.load_from_checkpoint(
        ckpt_path,
        model_1=model_1,
        model_2=model_2,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
    )

    trainer = L.Trainer()
    preds = trainer.predict(lit_model, test_loader)
    preds = np.concatenate(preds, axis=0)

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Save PNGs"):
        Image.fromarray(preds[idx]).save(os.path.join(SUBMISSION_DIR, row["image"]), "PNG")

    os.makedirs(os.path.dirname(SUBMISSION_ZIP), exist_ok=True)
    with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in glob(os.path.join(SUBMISSION_DIR, "*.png")):
            zf.write(fp, arcname=os.path.basename(fp))

    print(f"Saved: {SUBMISSION_ZIP}")

if __name__ == "__main__":
    main()
