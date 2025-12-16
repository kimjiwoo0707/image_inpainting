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

## 📰 모델 설명

본 프로젝트에서는 **CBAM(Convolutional Block Attention Module)** 이 적용된 U-Net++ 기반 2단계 이미지 복원 모델을 제안한다. 본 모델은 손실 영역 복원과 색상화를 분리하여 수행함으로써, 구조적 일관성과 색상 자연도를 동시에 향상시키는 것을 목표로 한다.

<img width="776" height="103" alt="데이콘 색상화 unet++" src="https://github.com/user-attachments/assets/28f82bc0-aec0-4332-89c8-29de1f9f3111" />  

그림. 제안하는 2단계 이미지 복원 파이프라인.  


1단계에서는 손실된 흑백 이미지를 복원하고, 2단계에서는 복원된 구조를 기반으로 색상화를 수행한다.

---


### 주요 모델 구성
단일 모델로 구조 복원과 색상화를 동시에 수행할 경우, 손실 영역의 구조 정보와 색상 정보가 서로 간섭하여 복원 품질이 저하되는 문제가 발생한다. 이를 해결하기 위해 본 연구에서는 **구조 복원과 색상 복원을 분리한 2-Stage 파이프라인**을 설계하였다.

각 Stage의 U-Net++ Encoder 최상위 Feature Map에는 **CBAM(Convolutional Block Attention Module)** 을 적용하였다. 이를 통해 중요한 채널 및 공간 정보를 강조함으로써 손실 영역과 구조적으로 중요한 부분에 대한 복원 성능을 향상시켰다.

---

### Stage별 동작 설명
1) Stage 1: Gray Mask Restoration  
   Stage 1에서는 손실 영역이 포함된 흑백 이미지를 입력으로 받아, 손상된 구조 정보를 복원하는 데 집중한다. 이 단계에서는 명암 대비, 경계선, 객체의 형태와 같은 구조적 특징을 우선적으로 학습한다.

2) Residual Connection  
Stage 1의 출력은 입력 이미지에 Residual Connection을 통해 더해진다. 이를 통해 이미 정상적으로 존재하는 영역의 정보 손실을 방지하고, 모델이 손실 영역에만 집중하여 복원하도록 유도하였다.

3) Stage 2: Color Restoration  
Stage 2에서는 Stage 1에서 복원된 흑백 구조 정보를 기반으로 **색상화(Color Restoration)** 를 수행한다. 이 단계에서는 객체의 질감, 색상 분포, 전역적인 색 균형을 학습하여 자연스러운 RGB 이미지 복원을 목표로 한다.

---
### 모델 구성 요약  
• Backbone: U-Net++  
• Attention: CBAM (Encoder 최상위 Feature 적용)  
• Encoder: EfficientNet-B4  
• Stage 1: Gray Mask Restoration (1 → 1)  
• Stage 2: Color Restoration (1 → 3)  
• Learning Strategy: Two-stage restoration with residual connection  

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


