# image_inpainting
이미지 복원 기술은 손상되거나 결손된 부분을 복구하고, 흑백 이미지에 자연스러운 색을 입히는 기술로, 역사적 사진 복원, 영상 편집, 의료 이미지 복구 등 다양한 분야에서 중요하게 활용되고 있다.

손실된 이미지의 결손 부분을 복구하고, 흑백 이미지에 자연스러운 색을 입히는 AI 알고리즘 개발하여 이러한 복원 기술을 활용할 수 있는 Vision AI 알고리즘을 개발하는 것을 목표로 한다.

# 🌈 이미지 색상화 및 손실 부분 복원 AI 🎨

손상된 흑백 이미지를 복원하고 자연스러운 색을 입히는 AI 알고리즘!  
이 프로젝트는 **역사적 사진 복원**, **의료 이미지 복구**, **영상 편집** 등 다양한 분야에서 활용될 수 있는 Vision AI 기술을 개발하기 위해 만들어졌습니다.

---

## 📂 데이터셋

### 제공된 데이터셋
- **train_input**: 흑백 및 손상된 학습 이미지 (29,603장)
- **train_gt**: 원본 학습 이미지 (29,603장)
- **test_input**: 흑백 및 손상된 테스트 이미지 (100장)

### 주요 파일
- **train.csv**: 학습 데이터의 입력 이미지와 정답 이미지 경로 매핑
- **test.csv**: 테스트 데이터의 입력 이미지 경로
- **sample_submission.zip**: 제출 파일 형식 샘플

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
| 🎨 입력 이미지 | ✨ 복원 이미지 | 🌟 원본 이미지 |
|----------------|---------------|----------------|
| input          | restored      | gt             |

---

## 🗂️ 프로젝트 구조

```plaintext
project-name/
│
├── data/           # 데이터 관련 폴더
├── notebooks/      # 탐색 및 학습 노트북
├── models/         # 학습된 모델 및 체크포인트
├── src/            # 코드 파일
├── submission/     # 제출 파일 생성 폴더
├── requirements.txt # 필요한 패키지 목록
├── README.md       # 프로젝트 설명서
└── .gitignore      # Git 제외 설정 파일

---

## 🛠️ 사용된 기술
markdown
코드 복사
- PyTorch
- Lightning
- Transformers
- Segmentation Models PyTorch
- UMAP & HDBSCAN
- scikit-image

