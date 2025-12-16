## 이미지 색상화 및 손실 부분 복원 AI 경진대회

주관: 데이콘(DACON)  
성과: Private Leaderboard 기준 상위 5% (7 / 140팀)  

---
## 📰 대회 개요

본 대회는 손상된 흑백 이미지를 복원하고 자연스러운 색상을 예측하는
이미지 색상화(Colorization) 및 손실 영역 복원(Inpainting) 문제를 주제로 한
Vision AI 경진대회이다.

손실된 이미지의 결손 영역을 복구하는 동시에,
흑백 이미지에 대해 원본과 유사한 색 분포를 복원하는 알고리즘을 개발하는 것을 목표로 하며,
단순한 픽셀 복원이 아닌 구조적 일관성과 시각적 자연스러움이 함께 요구되는 고난도 이미지 복원 문제로 구성되었다.

본 대회는 역사적 사진 복원, 영상 편집, 의료 영상 복구 등
실제 활용도가 높은 이미지 복원 기술을 연구·개발하는 것을 목적으로 하였으며,
Vision AI 기반 이미지 생성 및 복원 모델의 설계 역량을 종합적으로 평가하는 대회이다.

---
## 📂 데이터셋 사용 안내

본 프로젝트는 **데이콘**에서 제공하는 공개 데이터를 사용합니다.  
데이터는 아래의 링크에서 다운로드할 수 있습니다:

- [👉 데이콘 대회 페이지 바로가기](https://dacon.io/competitions)

---

### 📥 데이터 다운로드 방법

1️⃣ **데이콘 대회 페이지에 접속하여 대회에 참여합니다.**  
2️⃣ **아래 데이터를 다운로드하세요.**
   - `train_input`: 흑백, 일부 손상된 학습용 PNG 이미지 (29,603장)
   - `train_gt`: 원본 학습용 PNG 이미지 (29,603장)
   - `test_input`: 흑백, 일부 손상된 평가용 PNG 이미지 (100장)
   - `train.csv`: 학습용 입력 이미지와 정답 이미지 경로 매핑
   - `test.csv`: 테스트용 입력 이미지 경로

3️⃣ **다운로드한 데이터를 아래와 같은 디렉토리 구조로 배치하세요:**

```plaintext
project-name/
├── data/
│   ├── raw/                  # 원본 데이터
│   │   ├── train_input/      # 손상된 흑백 학습 이미지
│   │   ├── train_gt/         # 원본 학습 이미지
│   │   ├── test_input/       # 손상된 흑백 평가 이미지
│   ├── processed/            # 전처리된 데이터 (사용자가 생성)
├── notebooks/                # 탐색 및 학습 노트북
├── src/                      # 코드 파일
├── submission/               # 제출 파일 생성 폴더

```
---

### 🗂️ processed/ 폴더 설명

`processed/` 폴더에는 전처리된 데이터가 포함되어 있습니다.  
이 데이터는 깃허브에 별도로 업로드되어 있으며, 아래 링크에서 다운로드할 수 있습니다:

- [👉 processed/ 데이터 다운로드](https://github.com/username/repository-name/tree/main/data/processed)

---

### ⚠️ 데이터 사용 주의사항

- 본 프로젝트에서 사용된 데이터는 **데이콘**의 대회 데이터를 기반으로 하며, 데이터의 저작권은 데이콘 및 데이터 제공자에게 있습니다.
- 데이터를 직접 배포할 수 없으므로, 데이터를 사용하려면 반드시 [데이콘 대회 페이지](https://dacon.io/competitions)에서 직접 다운로드해야 합니다.
- **전처리된 데이터(`processed/`)**는 학습에 사용된 중간 결과이며, 깃허브에 업로드되어 있으므로 아래 링크에서 직접 다운로드하여 활용하시면 됩니다:
  - [👉 processed/ 데이터 다운로드](https://github.com/username/repository-name/tree/main/data/processed)

 
---

## 🚀 실행 방법


# 1️⃣ 환경 설정
pip install -r requirements.txt

# 2️⃣ 데이터 전처리
python src/data_preprocessing.py

# 3️⃣ 모델 학습
python src/train.py

# 4️⃣ 모델 추론
python src/inference.py

# 5️⃣ 제출 파일 생성
결과는 submission/output.zip 형태로 저장됩니다.

---

## 🧠 모델 설명

본 프로젝트에서는 **CBAM (Convolutional Block Attention Module)**을 활용하여 복원 품질을 극대화한 **U-Net++** 모델을 사용했습니다.  
**CBAM**은 채널과 공간의 주의 메커니즘을 통해 중요한 정보를 강조하여, 이미지 복원 및 색상화의 성능을 획기적으로 향상시킵니다.

---

### 📌 주요 모델 구성
1️⃣ **CBAM 적용된 U-Net++ 아키텍처**  
   - Encoder의 최상위 Feature Map에 CBAM을 적용해 복원 품질 개선.  
   - 채널 및 공간 주의 기법을 결합하여 중요한 정보를 학습.  

2️⃣ **두 단계 모델 학습**  
   - **단계 1**: 손상된 흑백 이미지를 복원 (Gray Mask Restoration).  
   - **단계 2**: 복원된 흑백 이미지를 컬러 이미지로 변환 (Gray to Color Conversion).  

---

### ⚙️ 하이퍼파라미터
- **학습률**: `1e-4`  
- **배치 크기**: `8`  
- **에폭 수**: `50`  
- **최적화 기법**: AdamW Optimizer + Cosine Scheduler  
- **평가 지표**:  
  - SSIM (Structural Similarity Index)  
  - Masked SSIM  
  - 히스토그램 유사도  

---

## 📈 성능 향상 방법

본 프로젝트의 성능을 높이기 위해 아래와 같은 방법을 적용했습니다:

### 🛠️ 주요 개선 기법
1️⃣ **CBAM Attention 적용**  
   - 채널과 공간의 중요도를 학습하여 특징을 강조.  
   - U-Net++의 Encoder에 CBAM을 추가해 복원 품질을 대폭 향상.  

2️⃣ **Feature Embedding & 클러스터링**  
   - **CLIP 모델**을 사용해 이미지 특징을 Embedding으로 추출.  
   - **UMAP & HDBSCAN**으로 이상치를 제거하고 데이터 품질을 개선.  

3️⃣ **효율적인 데이터 전처리**  
   - 랜덤 다각형 손상 영역 생성으로 데이터 다양성을 확보.  
   - 전처리된 데이터를 사용해 모델 학습 안정성을 강화.  

4️⃣ **효과적인 학습 전략**  
   - Early Stopping 및 Model Checkpoint로 과적합 방지.  
   - K-Fold Cross Validation으로 데이터 분할 및 일반화 성능 강화.  

5️⃣ **SSIM 기반 손실 함수**  
   - Masked SSIM을 통해 손실 영역의 복원 품질을 최적화.  

---

### 🚀 적용 결과
- **SSIM**: 이미지 전체의 복원 품질 평가.  
- **Masked SSIM**: 손실 영역에서의 복원 품질 평가.  
- **히스토그램 유사도**: 복원된 이미지의 색상 일치도 평가.


---


## 📊 주요 결과

### 모델 평가 지표
- **SSIM (Structural Similarity Index)**: 복원 이미지의 품질을 정량적으로 평가.
- **Masked SSIM**: 손실 영역에서의 복원 품질을 평가.
- **히스토그램 유사도**: 복원된 이미지의 색상 일치도를 평가.

---

### 시각적 비교 (예시)

| 🎨 입력 이미지         | ✨ 복원 이미지       |
|---------------------|-------------------|
| ![Input Image](https://github.com/user-attachments/assets/6f92ee36-ff94-4aad-97ca-78fe77e36ce8) | ![Restored Image](https://github.com/user-attachments/assets/ff9d9fe7-bda8-4074-af35-ce06c56237f8) |


---


## 🗂️ 프로젝트 구조

```plaintext
project-name/
├── data/                     # 데이터 관련 폴더
│   ├── raw/                  # 원본 데이터
│   ├── processed/            # 전처리된 데이터
├── notebooks/                # 탐색 및 학습 노트북
├── models/                   # 학습된 모델 및 체크포인트
├── src/                      # 코드 파일
│   ├── data_preprocessing.py # 데이터 전처리 코드
│   ├── train.py              # 학습 코드
│   ├── inference.py          # 추론 코드
├── submission/               # 제출 파일 생성 폴더
├── requirements.txt          # 필요한 패키지 목록
├── README.md                 # 프로젝트 설명서
└── .gitignore                # Git 제외 설정 파일
```

---


---

## 🛠️ 사용된 기술

### 주요 라이브러리
- **PyTorch**: 딥러닝 모델 구현 및 학습을 위한 프레임워크
- **Lightning**: PyTorch 기반의 학습 루프와 실험 관리 도구
- **Transformers**: 자연어 처리 및 컴퓨터 비전 모델을 위한 라이브러리
- **Segmentation Models PyTorch**: 이미지 분할 모델 구현을 위한 PyTorch 확장 라이브러리
- **UMAP & HDBSCAN**: 차원 축소 및 클러스터링 알고리즘
- **scikit-image**: 이미지 처리 및 분석을 위한 라이브러리


