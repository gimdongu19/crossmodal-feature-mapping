# PROMPTS.md

## 프로젝트명

Codex를 활용한 Crossmodal Feature Mapping 기반 멀티모달 산업 이상 탐지 오픈소스 구현 및 분석

## 사용한 AI 코딩 툴

OpenAI Codex

## 대상 논문

Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping

## 프로젝트 목적

본 프로젝트는 AI 코딩 도구를 활용하여 논문 공식 오픈소스 코드의 구조를 분석하고, 실행 매뉴얼과 보고서를 작성하는 것을 목표로 한다. 로컬 저장공간과 GPU 자원 제약으로 인해 전체 MVTec 3D-AD 데이터셋 학습은 수행하지 않았으며, 코드 분석 및 핵심 구현 흐름 이해를 중심으로 진행하였다.

---

## 1. 저장소 구조 분석

### Prompt

나는 캡스톤디자인 과제로 "Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping" 논문의 오픈소스 코드를 분석하려고 한다.

나는 Codex를 처음 사용한다. 로컬 저장공간이 부족하므로 전체 MVTec 3D-AD 데이터셋 다운로드나 전체 학습은 수행하지 않는다.

이번 작업에서는 코드를 수정하지 말고 분석만 수행해줘.

다음 내용을 한국어로 정리해줘.

1. 저장소 전체 구조
2. 학습 코드가 시작되는 파일
3. 추론 코드가 시작되는 파일
4. 2D feature extractor가 사용되는 위치
5. 3D feature extractor가 사용되는 위치
6. crossmodal feature mapping network가 구현된 위치
7. anomaly map이 생성되는 코드 흐름
8. MVTec 3D-AD 데이터셋을 실행하려면 필요한 폴더 구조
9. 실제 실행 시 예상되는 GPU, 저장공간, dependency 문제
10. 이 내용을 바탕으로 보고서에 넣을 수 있는 요약문

### Codex 답변 요약

Codex는 저장소의 주요 파일과 폴더 구조를 분석하였다. 학습은 `cfm_training.py`, 추론은 `cfm_inference.py`를 중심으로 수행되며, `models/`, `processing/`, `utils/` 폴더가 각각 모델 정의, 데이터 처리, 평가 및 시각화 보조 기능을 담당하는 것으로 정리하였다.

### 실제 반영 내용

보고서의 "오픈소스 코드 구조 분석" 파트에 반영하였다.

---

## 2. 학습 및 추론 실행 방법 분석

### Prompt

이 저장소에서 MVTec 3D-AD 데이터셋으로 학습과 추론을 실행하려면 어떤 파일을 사용해야 하는지 분석해줘. README와 shell script를 확인해서 학습 명령어, 추론 명령어, 필요한 인자, dataset path, checkpoint path, 결과 저장 경로를 정리해줘.

### Codex 답변 요약

Codex는 MVTec 3D-AD 학습을 위해 `01_train_mvtec.sh`와 `cfm_training.py`를 확인해야 하며, 추론 및 평가를 위해 `02_eval_mvtec.sh`와 `cfm_inference.py`를 확인해야 한다고 설명하였다. 또한 `--dataset_path`, `--checkpoint_savepath`, `--checkpoint_folder`, `--class_name`, `--epochs_no`, `--batch_size`, `--qualitative_folder`, `--quantitative_folder` 등의 인자가 중요하다고 정리하였다.

### 실제 반영 내용

보고서의 "실행 매뉴얼" 파트에 반영하였다.

---

## 3. Crossmodal Feature Mapping 구현 흐름 분석

### Prompt

Crossmodal Feature Mapping의 구현 흐름을 보고서에 넣을 수 있도록 한국어로 설명해줘. 2D feature extractor, 3D feature extractor, 2D-to-3D mapping, 3D-to-2D mapping, discrepancy 계산, product aggregation 순서로 설명해줘.

### Codex 답변 요약

Codex는 RGB 이미지와 point cloud에서 각각 frozen feature extractor를 통해 feature를 추출한 뒤, 두 개의 MLP mapping network를 이용해 서로 다른 modality의 feature를 예측하는 구조라고 설명하였다. 이후 관측 feature와 예측 feature 사이의 discrepancy를 계산하고, 2D anomaly map과 3D anomaly map을 pixel-wise product 방식으로 결합하여 최종 anomaly map을 생성한다고 정리하였다.

### 실제 반영 내용

보고서의 "Crossmodal Feature Mapping 구현 흐름" 파트에 반영하였다.

---

## 4. 실행 환경 및 제약 분석

### Prompt

이 프로젝트를 실제로 실행할 때 필요한 환경과 예상되는 문제를 정리해줘. 특히 GPU, 저장공간, MVTec 3D-AD 데이터셋, PyTorch/CUDA dependency, wandb 설정 문제를 중심으로 설명해줘. 또한 로컬 저장공간이 부족한 경우 과제 보고서에서 어떤 방식으로 실험 제약을 설명하면 좋을지 알려줘.

### Codex 답변 요약

Codex는 이 프로젝트가 RGB와 3D point cloud 데이터를 함께 사용하기 때문에 데이터셋 용량이 크고, DINO ViT와 Point-MAE 기반 feature extractor를 사용하므로 GPU 메모리와 실행 시간이 필요하다고 설명하였다. 또한 PyTorch/CUDA 버전 충돌, dataset path 오류, checkpoint path 오류, wandb 로그인 문제가 발생할 수 있다고 정리하였다.

### 실제 반영 내용

보고서의 "데이터셋 및 실험 제약"과 "한계점 및 개선 방향" 파트에 반영하였다.

---

## 5. 실행 매뉴얼 작성

### Prompt

보고서에 포함할 실행 매뉴얼을 작성해줘. 저장소 clone, conda 환경 생성, requirements 설치, MVTec 3D-AD 데이터셋 경로 설정, 학습 실행, 추론 실행, 결과 확인 순서로 작성해줘. 실제 전체 학습은 수행하지 않았으므로, 명령어는 예시로 표시하고 저장공간 및 GPU 제약으로 전체 재현은 생략했다고 명확히 써줘.

### Codex 답변 요약

Codex는 저장소 clone부터 Python 환경 구성, dependency 설치, dataset path 설정, `cfm_training.py` 기반 학습, `cfm_inference.py` 기반 추론, 결과 폴더 확인까지의 절차를 정리하였다. 또한 전체 학습을 수행하지 않은 이유를 저장공간과 GPU 제약으로 설명하도록 제안하였다.

### 실제 반영 내용

보고서의 "실행 매뉴얼" 파트에 반영하였다.

---

## 6. Toy demo 설계

### Prompt

실제 MVTec 3D-AD 데이터셋 없이 Crossmodal Feature Mapping의 핵심 아이디어를 보여주는 toy demo를 설계해줘. PyTorch random tensor를 2D feature와 3D feature처럼 사용하고, 2D-to-3D MLP와 3D-to-2D MLP를 학습한 뒤, discrepancy와 product aggregation으로 anomaly score를 계산하는 구조로 설명해줘. 이 toy demo는 논문 성능 재현이 아니라 방법론 이해를 위한 간소화 예제임을 명확히 적어줘.

### Codex 답변 요약

Codex는 random tensor를 사용하여 2D feature와 3D feature를 가정하고, 두 개의 MLP mapping network를 학습한 뒤, 관측 feature와 예측 feature 사이의 차이를 계산하여 anomaly score를 만드는 간소화 실험 구조를 제안하였다.

### 실제 반영 내용

보고서의 "간소화 구현 방향" 파트에 반영하였다.

---

## 7. 보고서 초안 작성

### Prompt

지금까지 분석한 내용을 바탕으로 한국어 보고서 초안을 작성해줘.

보고서 제목은 "Codex를 활용한 Crossmodal Feature Mapping 기반 멀티모달 산업 이상 탐지 오픈소스 구현 및 분석"으로 해줘.

보고서 목차는 다음과 같이 구성해줘.

1. 프로젝트 개요
2. 대상 논문 소개
3. AI 코딩 도구 선정 이유
4. 오픈소스 코드 구조 분석
5. Crossmodal Feature Mapping 구현 흐름
6. 실행 매뉴얼
7. 데이터셋 및 실험 제약
8. 간소화 구현 방향
9. 예상 결과 및 anomaly map 해석
10. 한계점 및 개선 방향
11. 결론
12. 참고 자료

주의사항:
- 전체 데이터셋 학습을 실제로 수행한 것처럼 쓰지 마.
- 코드 분석 및 실행 매뉴얼 중심으로 수행했다고 써.
- 논문의 핵심 구조인 2D-to-3D mapping, 3D-to-2D mapping, discrepancy calculation, product aggregation을 자세히 설명해.
- 대학 과제 보고서 문체로 작성해.

### Codex 답변 요약

Codex는 논문 방법론, 코드 구조, 실행 절차, 실험 제약, 간소화 구현 방향, 한계점 및 개선 방향을 포함한 보고서 초안을 작성하였다.

### 실제 반영 내용

`REPORT.md` 파일로 정리하였다.

---

## 8. PROMPTS.md 작성

### Prompt

지금까지의 Codex 활용 내용을 바탕으로 PROMPTS.md 파일을 작성해줘. 각 항목에는 Prompt, Codex 답변 요약, 실제 반영 내용을 포함해줘. 실제 수행하지 않은 학습이나 실험은 했다고 쓰지 말고, 전체 학습은 저장공간과 GPU 제약으로 수행하지 않았다고 명확히 써줘.

### Codex 답변 요약

Codex는 프로젝트 수행 중 사용한 프롬프트를 항목별로 정리하고, 각 프롬프트에 대한 답변 요약과 보고서 반영 내용을 정리하는 형식을 제안하였다.

### 실제 반영 내용

본 `PROMPTS.md` 파일로 정리하였다.

---

## 최종 정리

본 프로젝트에서는 Codex를 논문 오픈소스 코드 분석 보조 도구로 활용하였다. 주요 활용 내용은 저장소 구조 분석, 학습 및 추론 흐름 파악, Crossmodal Feature Mapping 구현 원리 정리, 실행 매뉴얼 작성, toy demo 설계, 보고서 초안 작성이다. 전체 MVTec 3D-AD 데이터셋 학습은 저장공간과 GPU 제약으로 수행하지 않았으며, 과제에서는 코드 분석 및 문서화 중심으로 결과를 정리하였다.
