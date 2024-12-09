import os
import numpy as np
import pandas as pd
import torch
import lightning as L
from segmentation_models_pytorch import Unet
from model import CBAMUnetPlusPlus, LitIRModel  # 필요한 모델 클래스 임포트
from torch.utils.data import DataLoader
from dataset import CustomImageDataset, CollateFn  # 데이터셋 클래스 임포트
from sklearn.model_selection import KFold
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# 하이퍼파라미터 설정
SEED = 42
N_SPLIT = 5
BATCH_SIZE = 8

L.seed_everything(SEED)

# 데이터 로드 및 K-Fold 분할
train_df = pd.read_csv('./data/train_preproc.csv')
kf = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)
train_indices, valid_indices = next(kf.split(train_df['image'], train_df['label']))
train_fold_df = train_df.iloc[train_indices].reset_index(drop=True)
valid_fold_df = train_df.iloc[valid_indices].reset_index(drop=True)

# 데이터셋 및 데이터로더 생성
train_dataset = CustomImageDataset(train_fold_df, data_dir='./data/raw/train_gt', mode='train')
valid_dataset = CustomImageDataset(valid_fold_df, data_dir='./data/raw/train_gt', mode='valid')

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=CollateFn(mode='train'))
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, collate_fn=CollateFn(mode='valid'))

# 모델 정의
model_1 = CBAMUnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights="imagenet", in_channels=1, classes=1)
model_2 = CBAMUnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights="imagenet", in_channels=1, classes=3)
lit_ir_model = LitIRModel(model_1=model_1, model_2=model_2)

# 콜백 설정
checkpoint_callback = ModelCheckpoint(
    monitor='val_score', mode='max', dirpath='./checkpoint/',
    filename='best-checkpoint', save_top_k=1, verbose=True
)
earlystopping_callback = EarlyStopping(monitor="val_score", mode="max", patience=7)

# 학습
trainer = L.Trainer(
    max_epochs=50,
    precision='bf16-mixed',
    callbacks=[checkpoint_callback, earlystopping_callback]
)
trainer.fit(lit_ir_model, train_dataloader, valid_dataloader)
