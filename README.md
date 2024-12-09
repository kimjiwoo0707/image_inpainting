# image_inpainting
이미지 복원 기술은 손상되거나 결손된 부분을 복구하고, 흑백 이미지에 자연스러운 색을 입히는 기술로, 역사적 사진 복원, 영상 편집, 의료 이미지 복구 등 다양한 분야에서 중요하게 활용되고 있다.

손실된 이미지의 결손 부분을 복구하고, 흑백 이미지에 자연스러운 색을 입히는 AI 알고리즘 개발하여 이러한 복원 기술을 활용할 수 있는 Vision AI 알고리즘을 개발하는 것을 목표로 한다.

# 🌈 이미지 색상화 및 손실 부분 복원 AI 🎨

손상된 흑백 이미지를 복원하고 자연스러운 색을 입히는 AI 알고리즘!  
이 프로젝트는 **역사적 사진 복원**, **의료 이미지 복구**, **영상 편집** 등 다양한 분야에서 활용될 수 있는 Vision AI 기술을 개발하기 위해 만들어졌습니다.

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

## 🧠 모델 구조

### 주요 구성 요소
1. **CBAM (Convolutional Block Attention Module)**  
   - 채널 및 공간 중요도를 동적으로 학습하여 복원 품질 향상.
2. **U-Net++ 아키텍처**  
   - 다층 연결 구조를 활용한 효율적인 복원.

### 하이퍼파라미터
- **학습률**: `1e-4`
- **배치 크기**: `8`
- **에폭 수**: `50`

---
# 🌈 이미지 색상화 및 손실 부분 복원 AI 🎨

손상된 흑백 이미지를 복원하고 자연스러운 색을 입히는 AI 알고리즘!  
이 프로젝트는 **역사적 사진 복원**, **의료 이미지 복구**, **영상 편집** 등 다양한 분야에서 활용될 수 있는 Vision AI 기술을 개발하기 위해 만들어졌습니다.

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

## 📊 주요 결과
markdown
코드 복사
### 모델 평가 지표
- SSIM (Structural Similarity Index): 복원 이미지 품질
- Masked SSIM: 손실 영역 복원 품질
- 히스토그램 유사도: 색상화 품질

---

## 📊 주요 결과

### 시각적 비교 (예시)
| 🎨 입력 이미지 | ✨ 복원 이미지 |
|----------------|---------------|
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


