# Codex를 활용한 Crossmodal Feature Mapping 기반 멀티모달 산업 이상 탐지 오픈소스 구현 및 분석

## 1. 프로젝트 개요

본 프로젝트는 AI 코딩 도구인 OpenAI Codex를 활용하여 논문 「Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping」의 오픈소스 구현 구조를 분석하고, 실행 매뉴얼과 핵심 구현 내용을 정리하는 것을 목표로 한다. 해당 논문은 산업 이상 탐지 문제에서 RGB 이미지와 3D point cloud 정보를 함께 활용하는 멀티모달 anomaly detection 방법을 제안한다.

본 과제의 핵심 목표는 단순히 논문 내용을 요약하는 것이 아니라, 실제 공개된 오픈소스 코드가 논문의 방법론을 어떤 구조로 구현하고 있는지 분석하는 것이다. 또한 AI 코딩 도구를 사용하여 코드 구조 파악, 실행 흐름 분석, 주요 모듈 설명, 실행 매뉴얼 작성, 간소화 구현 예제 설계 과정을 수행하였다.

다만 본 프로젝트에서는 로컬 저장공간과 GPU 자원의 제약으로 인해 MVTec 3D-AD 전체 데이터셋을 다운로드하여 전체 학습을 재현하지는 않았다. 대신 공식 코드 구조 분석과 실행 절차 정리, 그리고 핵심 아이디어를 설명하기 위한 간소화 구현 방향을 중심으로 과제를 수행하였다.

## 2. 대상 논문 소개

대상 논문은 RGB 이미지와 3D point cloud를 함께 사용하는 멀티모달 산업 이상 탐지 방법을 제안한다. 기존의 산업 이상 탐지 방법은 RGB 이미지 기반으로 많이 연구되었지만, 실제 산업 현장에서는 조명 변화, 표면 반사, 색상 변화 등으로 인해 RGB 정보만으로 결함을 안정적으로 탐지하기 어려운 경우가 있다. 반대로 3D 정보는 표면 형상이나 깊이 변화를 반영할 수 있으나, 색상이나 질감 기반의 결함을 충분히 설명하지 못할 수 있다. 따라서 두 modality를 함께 활용하는 방식이 필요하다.

논문에서 제안하는 핵심 아이디어는 정상 샘플에서 RGB feature와 3D feature 사이의 관계를 학습한 뒤, 테스트 시점에 관측된 feature와 crossmodal mapping으로 예측된 feature 사이의 불일치를 이용하여 이상 영역을 탐지하는 것이다. 즉, 정상 데이터에서는 2D feature와 3D feature 사이의 관계가 일관되게 나타난다고 가정하고, 이 관계에서 벗어나는 영역을 anomaly로 판단한다.

이 방법은 memory bank를 사용하는 기존 방법과 달리 모든 정상 feature를 저장하고 비교하지 않는다. 대신 2D feature에서 3D feature를 예측하는 MLP와 3D feature에서 2D feature를 예측하는 MLP를 학습한다. 이로 인해 메모리 사용량과 추론 시간을 줄이면서도 높은 anomaly detection 및 segmentation 성능을 목표로 한다.

## 3. AI 코딩 도구 선정 이유

본 프로젝트에서는 AI 코딩 도구로 OpenAI Codex를 선택하였다. Codex는 저장소의 코드 구조를 탐색하고, 주요 파일의 역할을 설명하며, 실행 명령어와 데이터셋 경로 설정 방법을 분석하는 데 활용할 수 있다. 특히 공개 오픈소스 코드 기반 과제에서는 다음과 같은 장점이 있다.

첫째, 복잡한 코드베이스의 구조를 빠르게 파악할 수 있다. 논문 구현 코드는 학습 코드, 추론 코드, 모델 코드, 데이터 처리 코드, 평가 코드가 여러 파일로 나뉘어 있기 때문에 처음 접근할 때 전체 흐름을 이해하기 어렵다. Codex를 활용하면 주요 entry point와 모듈 간 연결 관계를 빠르게 정리할 수 있다.

둘째, 실행 매뉴얼을 작성하는 데 유용하다. 공식 저장소에 README가 있더라도 실제 과제 보고서에서는 환경 구성, dependency 설치, 데이터셋 준비, 학습 및 추론 명령어를 한국어로 정리할 필요가 있다. Codex는 이러한 과정을 문서화하는 데 도움을 줄 수 있다.

셋째, 오류 해결 과정 기록에 활용할 수 있다. 실제 실행 과정에서는 CUDA 버전, PyTorch 버전, dataset path, checkpoint path, wandb 설정 등 다양한 문제가 발생할 수 있다. Codex는 오류 메시지를 분석하고 해결 방향을 제안하는 도구로 활용 가능하다.

## 4. 오픈소스 코드 구조 분석

공식 오픈소스 저장소는 Crossmodal Feature Mapping을 학습하고 추론하기 위한 코드로 구성되어 있다. 주요 파일과 폴더의 역할은 다음과 같이 정리할 수 있다.

| 구분 | 역할 |
|---|---|
| `cfm_training.py` | Crossmodal Feature Mapping 모델을 학습하는 entry point |
| `cfm_inference.py` | 학습된 CFM checkpoint를 이용하여 anomaly map과 평가 결과를 생성하는 entry point |
| `models/` | feature extractor 및 mapping network 등 모델 관련 코드 |
| `processing/` | RGB, point cloud, feature alignment, 데이터 전처리 관련 코드 |
| `utils/` | metric 계산, 결과 저장, 시각화 등 보조 함수 |
| `01_train_mvtec.sh` | MVTec 3D-AD 학습 실행 예시 스크립트 |
| `02_eval_mvtec.sh` | MVTec 3D-AD 추론 및 평가 실행 예시 스크립트 |
| `03_train_eyecandies.sh` | Eyecandies 데이터셋 학습 실행 예시 스크립트 |
| `requirements.txt` | 실행에 필요한 Python package 목록 |

학습은 `cfm_training.py`를 중심으로 진행된다. 이 파일에서는 데이터셋 경로, 클래스 이름, epoch 수, batch size, checkpoint 저장 경로 등을 인자로 받아 학습을 수행한다. 추론은 `cfm_inference.py`를 통해 수행되며, 학습된 checkpoint를 불러와 테스트 데이터에 대해 anomaly map과 정량 평가 결과를 생성한다.

공식 저장소에서는 MVTec 3D-AD 기준 학습 예시는 `01_train_mvtec.sh`, 평가 예시는 `02_eval_mvtec.sh`를 참고하도록 안내한다. 또한 학습 과정에서는 wandb를 사용하므로, wandb 계정이 없거나 사용하지 않을 경우 `cfm_training.py`에서 wandb 설정을 비활성화해야 한다.

## 5. Crossmodal Feature Mapping 구현 흐름

논문의 전체 구현 흐름은 크게 feature extraction, crossmodal feature mapping, discrepancy calculation, anomaly map aggregation으로 나눌 수 있다.

### 5.1 2D feature extraction

RGB 이미지는 frozen 2D feature extractor를 통과하여 2D feature map으로 변환된다. 논문에서는 ImageNet 기반으로 학습된 DINO ViT-B/8을 2D feature extractor로 사용한다. 이때 feature extractor는 학습 중 업데이트되지 않고 고정된 상태로 사용된다.

2D feature extractor의 출력은 원본 이미지보다 낮은 해상도의 feature map일 수 있으므로, 이후 anomaly map 생성을 위해 원본 이미지 해상도 또는 공통 feature map 해상도에 맞게 보간된다.

### 5.2 3D feature extraction

3D point cloud는 frozen 3D feature extractor를 통해 3D feature로 변환된다. 논문에서는 ShapeNet 기반으로 학습된 Point-MAE를 3D feature extractor로 사용한다. 3D feature 역시 학습 과정에서 업데이트되지 않고 고정된다.

3D feature는 point cloud의 일부 중심점 또는 group 단위로 계산될 수 있으므로, 이후 각 point 또는 pixel 위치에 대응되도록 interpolation과 alignment 과정이 필요하다. MVTec 3D-AD 데이터셋은 RGB 이미지와 3D 정보가 pixel-registered 형태로 제공되기 때문에 2D feature와 3D feature를 같은 위치 기준으로 비교할 수 있다.

### 5.3 Crossmodal mapping network

Crossmodal Feature Mapping의 핵심은 두 개의 mapping network이다.

첫 번째 mapping network는 2D feature를 입력받아 3D feature를 예측한다. 즉, RGB 이미지에서 추출된 feature가 주어졌을 때 해당 위치의 3D feature가 어떻게 나타나야 하는지를 학습한다.

두 번째 mapping network는 3D feature를 입력받아 2D feature를 예측한다. 즉, point cloud에서 추출된 feature가 주어졌을 때 해당 위치의 RGB feature가 어떻게 나타나야 하는지를 학습한다.

두 mapping network는 lightweight MLP 구조로 구성되며, 정상 샘플만을 사용해 학습된다. 학습 loss는 관측 feature와 예측 feature 사이의 cosine distance를 최소화하는 방식으로 구성된다.

### 5.4 Discrepancy calculation

테스트 단계에서는 먼저 실제 RGB 이미지와 point cloud에서 각각 2D feature와 3D feature를 추출한다. 이후 학습된 mapping network를 통해 2D feature로부터 예측된 3D feature, 3D feature로부터 예측된 2D feature를 얻는다.

정상 영역에서는 관측 feature와 예측 feature가 유사해야 한다. 반면 anomaly 영역에서는 정상 데이터에서 학습한 crossmodal 관계와 다른 패턴이 나타나므로, 관측 feature와 예측 feature 사이의 차이가 커진다. 이 차이를 이용해 2D anomaly map과 3D anomaly map을 각각 계산한다.

### 5.5 Product aggregation

논문에서는 2D anomaly map과 3D anomaly map을 결합하기 위해 pixel-wise product를 사용한다. 이는 두 modality에서 모두 이상 신호가 확인되는 위치를 더 강하게 anomaly로 판단하는 방식이다.

즉, 최종 anomaly map은 다음과 같은 의미를 가진다.

`Final anomaly map = 2D discrepancy map × 3D discrepancy map`

이 방식은 단일 modality에서만 발생하는 불확실한 false positive를 줄이는 데 도움을 줄 수 있다. RGB와 3D feature 간 관계가 동시에 불일치하는 위치를 anomaly로 판단하기 때문이다.

## 6. 실행 매뉴얼

본 절에서는 공식 오픈소스 코드를 실행하기 위한 절차를 정리한다. 실제 전체 학습은 저장공간과 GPU 제약으로 수행하지 않았으며, 아래 내용은 공식 저장소 구조 분석을 바탕으로 작성한 실행 매뉴얼이다.

### 6.1 저장소 준비

```bash
git clone https://github.com/CVLAB-Unibo/crossmodal-feature-mapping.git
cd crossmodal-feature-mapping
```

과제용으로는 원본 저장소를 fork한 뒤, `REPORT.md`, `PROMPTS.md`, `demo/` 폴더 등을 추가하는 방식이 적절하다.

### 6.2 Python 환경 구성

```bash
conda create -n cfm python=3.9 -y
conda activate cfm
pip install -r requirements.txt
```

다만 PyTorch, CUDA, point cloud 관련 package는 실행 환경에 따라 충돌이 발생할 수 있으므로, 설치 오류가 발생하면 CUDA 버전과 PyTorch 버전을 먼저 확인해야 한다.

### 6.3 데이터셋 준비

MVTec 3D-AD 데이터셋은 RGB 이미지와 3D 정보가 함께 제공되는 multimodal dataset이다. 코드 실행을 위해서는 dataset root path를 지정해야 한다. 예시 구조는 다음과 같다.

```text
datasets/
└── mvtec_3d_ad/
    ├── bagel/
    ├── cable_gland/
    ├── carrot/
    ├── cookie/
    ├── dowel/
    ├── foam/
    ├── peach/
    ├── potato/
    ├── rope/
    └── tire/
```

각 class 폴더 내부에는 train, test, ground truth 및 RGB/3D 관련 데이터가 코드에서 요구하는 형식으로 존재해야 한다.

### 6.4 학습 실행

MVTec 3D-AD 학습은 `01_train_mvtec.sh` 또는 `cfm_training.py`를 통해 수행한다.

예시:

```bash
bash 01_train_mvtec.sh
```

또는 Python script를 직접 실행하는 경우 다음과 같은 인자를 설정해야 한다.

```bash
python cfm_training.py \
  --dataset_path /path/to/mvtec_3d_ad \
  --checkpoint_savepath checkpoints/checkpoints_CFM_mvtec \
  --class_name bagel \
  --epochs_no 250 \
  --batch_size 1
```

실제 실행 시 class 이름, epoch 수, batch size, checkpoint 저장 경로는 환경에 맞게 수정해야 한다.

### 6.5 추론 및 anomaly map 생성

추론은 `cfm_inference.py` 또는 `02_eval_mvtec.sh`를 통해 수행한다.

예시:

```bash
bash 02_eval_mvtec.sh
```

또는 Python script를 직접 실행하는 경우 다음과 같은 인자를 설정할 수 있다.

```bash
python cfm_inference.py \
  --dataset_path /path/to/mvtec_3d_ad \
  --checkpoint_folder checkpoints/checkpoints_CFM_mvtec \
  --class_name bagel \
  --epochs_no 250 \
  --batch_size 1 \
  --qualitative_folder results/qualitative \
  --quantitative_folder results/quantitative \
  --produce_qualitatives
```

추론 결과로는 anomaly map 시각화 결과와 I-AUROC, P-AUROC, AUPRO 등의 정량 평가 결과가 저장된다.

## 7. 데이터셋 및 실험 제약

본 프로젝트에서는 전체 MVTec 3D-AD 데이터셋을 로컬 환경에 다운로드하여 학습을 수행하지 않았다. 그 이유는 다음과 같다.

첫째, MVTec 3D-AD는 RGB와 3D 데이터를 함께 포함하기 때문에 일반적인 RGB image dataset보다 저장공간 부담이 크다. 둘째, DINO ViT와 Point-MAE 기반 feature extractor를 사용하므로 GPU 메모리와 실행 시간이 많이 요구된다. 셋째, 공식 성능을 재현하려면 동일한 preprocessing, checkpoint, dependency, hardware 환경이 필요하지만, 개인 로컬 환경에서는 이를 모두 맞추기 어렵다.

따라서 본 과제에서는 전체 성능 재현보다 코드 구조 분석, 실행 조건 파악, 실행 매뉴얼 작성, 핵심 아이디어의 간소화 구현 설계에 초점을 두었다.

## 8. 간소화 구현 방향

전체 데이터셋을 사용하지 않더라도 논문의 핵심 아이디어를 이해하기 위해 toy demo를 구성할 수 있다. toy demo에서는 실제 RGB 이미지와 point cloud 대신 random tensor를 사용하여 2D feature와 3D feature를 가정한다.

간소화 구현의 흐름은 다음과 같다.

1. random tensor로 2D feature와 3D feature 생성
2. MLP 기반 2D-to-3D mapping network 정의
3. MLP 기반 3D-to-2D mapping network 정의
4. 정상 feature 쌍을 이용하여 cosine distance loss로 학습
5. 테스트 feature에 인위적인 anomaly pattern 추가
6. 관측 feature와 예측 feature 사이의 discrepancy 계산
7. 2D anomaly score와 3D anomaly score를 곱해 최종 anomaly score 생성

이 toy demo는 논문 성능을 재현하는 목적이 아니라, crossmodal feature mapping이 어떤 방식으로 anomaly score를 생성하는지 이해하기 위한 교육용 예제이다.

## 9. 예상 결과 및 anomaly map 해석

정상 샘플의 경우 2D feature와 3D feature 사이의 관계가 학습 데이터와 유사하기 때문에 mapping network가 예측한 feature와 실제 feature 사이의 차이가 작게 나타난다. 따라서 anomaly score가 낮게 형성된다.

반면 비정상 샘플에서는 RGB 또는 3D 형상 중 하나 또는 둘 모두에서 정상 데이터와 다른 패턴이 나타날 수 있다. 이 경우 2D-to-3D 또는 3D-to-2D mapping 결과가 실제 feature와 불일치하게 되고, discrepancy가 증가한다. 최종 anomaly map에서는 이러한 불일치가 큰 위치가 강조된다.

논문에서는 product aggregation을 사용하여 2D anomaly map과 3D anomaly map을 결합한다. 이 방식은 두 modality에서 모두 이상 징후가 확인되는 영역을 강하게 반영하므로, anomaly localization 성능을 높이는 데 기여한다.

## 10. 한계점 및 개선 방향

본 프로젝트의 한계는 다음과 같다.

첫째, 전체 데이터셋 학습과 정량 성능 재현을 수행하지 못했다. 따라서 논문에서 보고한 I-AUROC, P-AUROC, AUPRO 성능을 직접 재현하지는 못하였다.

둘째, 실제 산업 환경에서 사용하려면 RGB와 3D 정보가 정확히 정렬되어야 한다. 2D-3D alignment가 부정확하면 feature discrepancy가 anomaly 때문인지 정렬 오류 때문인지 구분하기 어렵다.

셋째, 해당 방법은 multimodal-only 구조이므로 RGB 데이터만 있거나 3D 데이터만 있는 환경에는 바로 적용하기 어렵다.

향후 개선 방향으로는 다음을 고려할 수 있다.

1. MVTec 3D-AD 일부 class만 사용한 최소 실행 실험
2. pretrained checkpoint 기반 inference만 수행하여 anomaly map 시각화
3. layer pruning 적용 여부에 따른 속도와 메모리 사용량 비교
4. depth map 또는 surface normal map을 이용한 간소화된 2D 기반 multimodal 실험
5. product aggregation 외 sum, max aggregation과의 비교 실험

## 11. 결론

본 프로젝트는 Codex를 활용하여 Crossmodal Feature Mapping 기반 멀티모달 산업 이상 탐지 오픈소스 코드를 분석하고, 실행 매뉴얼과 보고서 형태로 정리한 과제이다. 대상 논문은 RGB 이미지와 3D point cloud feature 사이의 관계를 학습하고, 관측 feature와 예측 feature 간 discrepancy를 이용하여 anomaly map을 생성하는 효율적인 방법을 제안한다.

본 과제에서는 전체 학습 재현보다 코드 구조 분석과 방법론 이해에 초점을 두었다. 이를 통해 학습 및 추론 entry point, feature extractor와 mapping network의 역할, anomaly map 생성 흐름, 실행 환경 및 데이터셋 요구사항을 정리하였다. 또한 PROMPTS.md를 통해 AI 코딩 도구 활용 과정을 기록하였다.

결과적으로 본 프로젝트는 AI 코딩 도구가 논문 오픈소스 분석, 실행 매뉴얼 작성, 코드 구조 이해, 보고서 작성에 어떻게 활용될 수 있는지를 보여준다.

## 12. 참고 자료

- Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping, uploaded paper.
- CVLAB-Unibo/crossmodal-feature-mapping official GitHub repository.
