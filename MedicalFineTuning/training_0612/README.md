# LLM Fine-tuning with LoRA/QLoRA

대형 언어 모델(LLM) 파인튜닝을 위한 LoRA/QLoRA 기반 훈련 스크립트입니다.

---

## 🔧 주요 특징

- **LoRA/QLoRA 지원**: 메모리 효율적인 파인튜닝
- **DeepSpeed 통합**: 분산 훈련 최적화  
- **4bit 양자화**: QLoRA를 통한 메모리 절약
- **자동 체크포인트**: 훈련 중단 시 재시작 가능

---

## 📦 요구사항

```bash
pip install torch transformers datasets peft accelerate deepspeed bitsandbytes wandb
```

---

## 📄 데이터셋 형식

데이터셋은 다음 형식이어야 합니다:

```json
{
  "input": "질문 텍스트",
  "output": "답변 텍스트", 
  "source": "데이터 출처 (선택사항)"
}
```

---

## ⚙️ 사용법

### 1. 설정 수정

`run.sh`에서 경로와 하이퍼파라미터를 수정하세요:

```bash
# 모델 및 데이터 경로
BASE_MODEL_NAME="/path/to/your/base/model"
DATA_PATH="/path/to/your/dataset"
CHECKPOINT_PATH="/path/to/checkpoint"
BASE_SAVE_PATH="/path/to/save/model"

# 하이퍼파라미터
USE_QLORA="True"  # QLoRA 사용 여부
NUM_EPOCHS=3      # 훈련 에폭 수
LORA_RANK=16      # LoRA 랭크
```

---

### 2. 훈련 실행

```bash
chmod +x run.sh
./run.sh
```

`run.sh` 내부에서 `train.py`가 호출되며, 아래 기능을 수행합니다:

#### 🔹 `train.py`: 파인튜닝 실행 스크립트

- Hugging Face의 `transformers` 및 `peft` 라이브러리를 기반으로 LoRA/QLoRA 파인튜닝
- `Trainer` 또는 `SFTTrainer` 사용
- QLoRA 사용 시 4bit 양자화 적용
- `wandb` 연동을 통한 로깅 지원

```bash
python train.py \
    --base_model "/path/to/base_model" \
    --dataset_path "/path/to/dataset" \
    --output_dir "/path/to/save/output" \
    --use_qlora True \
    --num_epochs 3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --learning_rate 1e-4
```

---

### 3. 훈련 재시작 (옵션)

체크포인트에서 훈련을 재시작하려면:

```bash
# run.sh에서 JOB 변수 수정
JOB="resume_training"
```

---

## 🔗 모델 병합 및 추론 준비

훈련이 완료된 후 어댑터 모델을 베이스 모델과 병합하여 추론 최적화 모델을 생성할 수 있습니다.

#### 🔹 `model_merge.py`: LoRA 어댑터 병합 스크립트

- LoRA/QLoRA 어댑터를 base model에 병합
- 최종 모델을 Hugging Face 호환 포맷으로 저장

```bash
python model_merge.py \
    --base_model "/path/to/base_model" \
    --adapter_model "/path/to/adapter" \
    --output_dir "/path/to/merged_model"
```

---

## ⚙️ 주요 파라미터

| 파라미터        | 설명                           | 기본값                     |
|------------------|----------------------------------|-----------------------------|
| `USE_QLORA`      | QLoRA 사용 여부                  | `True`                      |
| `NUM_EPOCHS`     | 훈련 에폭 수                     | `3`                         |
| `LORA_RANK`      | LoRA 랭크 (모델 복잡도)         | `16`                        |
| `LORA_ALPHA`     | LoRA 스케일링 파라미터           | `32`                        |
| `LEARNING_RATE`  | 학습률                            | `1e-4` (QLoRA), `5e-6` (LoRA) |

---

## 🧠 메모리 최적화 전략

- **QLoRA**: 4bit 양자화로 메모리 사용량 대폭 감소  
- **Gradient Checkpointing**: 메모리 ↔ 계산 트레이드오프  
- **DeepSpeed**: 분산 훈련으로 메모리 분산  

---

## 📊 모니터링

훈련 진행 상황은 **Weights & Biases (wandb)** 로 모니터링됩니다:

- **프로젝트명**: `RAPA`  
- **실행명**: `{모델명}_{LoRA유형}`  

---

## 📁 출력 구조

```plaintext
final_model/
├── {model_name}/
│   └── {num_epochs}/
│       ├── qlora_adapters/  # QLoRA 어댑터
│       └── qlora_merged/   # QLoRA 병합된 모델
│       └── lora_adapters/   # LoRA 어댑터
│       └── lora_merged/   # LoRA 병합된 모델

```
