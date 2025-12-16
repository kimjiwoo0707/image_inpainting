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

## 📰 데이터셋 설명

본 프로젝트는 데이콘 이미지 색상화 및 손실 부분 복원 AI 경진대회에서 제공한 공개 데이터셋을 사용하였다. 데이터는 손상된 흑백 이미지와 원본 컬러 이미지 쌍(pair) 으로 구성되어 있으며, 이미지 복원과 색상화를 동시에 수행하는 문제로 설계되었다.

| 데이터셋          | 설명                   | 개수     |
| ------------- | -------------------- | ------ |
| train_input | 손실 영역이 포함된 흑백 입력 이미지 | 29,603 |
| `train_gt`    | 원본 컬러 정답 이미지         | 29,603 |
| `test_input`  | 손실된 흑백 테스트 이미지       | 100    |
| `train.csv`   | 학습용 입력–정답 이미지 매핑 정보  | -      |
| `test.csv`    | 테스트 이미지 경로 정보        | -      |

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

### 📰 학습 설정 및 평가 지표

모델의 안정적인 학습과 복원 품질 향상을 위해
다음과 같은 학습 설정을 적용하였다.

- **Optimizer**: AdamW  
  → 가중치 감쇠를 통해 일반화 성능 향상

- **Learning Rate**: 1e-4  
- **Scheduler**: Cosine Annealing  
  → 학습 후반부 진동을 줄이고 안정적인 수렴 유도

- **Batch Size**: 8  
- **Epochs**: 50 (Early Stopping 적용)

---

## 📰 성능 향상 전략

모델 성능 및 복원 품질을 향상시키기 위해 아래와 같은 주요 전략을 적용하였다.

### 핵심 개선 요소

- **Attention 강화 (CBAM)**  
  → 중요한 채널 및 공간 정보를 강조하여 손실 영역 복원 성능 향상

- **데이터 품질 개선**  
  → CLIP 기반 Feature Embedding 후 UMAP & HDBSCAN을 활용해 이상치 제거

- **손실 영역 중심 학습**  
  → Masked SSIM 기반 손실 함수로 손실 영역 복원 품질 최적화

- **안정적인 학습 전략**  
  → Early Stopping, Model Checkpoint, K-Fold Cross Validation 적용


---


## 📰 평가 방식

본 대회는 이미지 복원 및 색상화 성능을 정량적으로 평가하기 위해 **SSIM 기반 복합 점수**를 사용한다.

### 평가 산식

최종 점수는 다음 세 가지 지표의 가중 평균으로 계산된다.

- S (SSIM): 전체 이미지에 대한 구조적 유사도 평균

- M (Masked SSIM): 손실(마스킹) 영역에 대한 SSIM 평균

- C (Color Histogram Similarity): 색상 히스토그램 기반 유사도 평균
  
<img width="369" height="34" alt="image" src="https://github.com/user-attachments/assets/096a2d88-9a67-4418-a036-7ca7d935804e" />  

손실 영역(M)과 색상 유사도(C)에 더 높은 가중치를 부여하여, 단순 구조 복원뿐 아니라 손실 영역 복원 품질과 색상 자연스러움을 중점적으로 평가한다.

---

### 시각적 비교 (예시)

|입력 이미지         |복원 이미지       |
|---------------------|-------------------|
| ![Input Image](https://github.com/user-attachments/assets/6f92ee36-ff94-4aad-97ca-78fe77e36ce8) | ![Restored Image](https://github.com/user-attachments/assets/ff9d9fe7-bda8-4074-af35-ce06c56237f8) |

