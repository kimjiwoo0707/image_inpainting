import os
import numpy as np
from glob import glob
from PIL import Image
from model import CBAMUnetPlusPlus, LitIRModel  # 필요한 모델 클래스 임포트
from dataset import CustomImageDataset, CollateFn  # 데이터셋 클래스 임포트
from torch.utils.data import DataLoader
import zipfile
import torch

# 하이퍼파라미터 및 경로 설정
SUBMISSION_DIR = './submission/'
SUBMISSION_FILE = './submission/output.zip'
TEST_DATA_DIR = './data/raw/test_input'
CHECKPOINT_PATH = './checkpoint/best-checkpoint.ckpt'

# 모델 로드
model_1 = CBAMUnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights=None, in_channels=1, classes=1)
model_2 = CBAMUnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights=None, in_channels=1, classes=3)
lit_ir_model = LitIRModel.load_from_checkpoint(CHECKPOINT_PATH, model_1=model_1, model_2=model_2)

# 테스트 데이터 로드
test_df = pd.read_csv('./data/test_preproc.csv')
test_dataset = CustomImageDataset(test_df, data_dir=TEST_DATA_DIR, mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=CollateFn(mode='test'))

# 추론
trainer = torch.hub.load('pytorch_lightning', 'trainer')
predictions = trainer.predict(lit_ir_model, test_dataloader)

# 결과 저장
os.makedirs(SUBMISSION_DIR, exist_ok=True)
for idx, row in test_df.iterrows():
    image_pred = Image.fromarray(predictions[idx])
    image_pred.save(os.path.join(SUBMISSION_DIR, row['image']), "PNG")

with zipfile.ZipFile(SUBMISSION_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_path in glob(f"{SUBMISSION_DIR}/*.png"):
        arcname = os.path.relpath(file_path, SUBMISSION_DIR)
        zipf.write(file_path, arcname)
