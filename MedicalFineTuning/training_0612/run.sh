#!/bin/bash

# =============================================================================
# 대형 언어 모델 파인튜닝 스크립트 (LoRA/QLoRA)
# PyTorch + DeepSpeed + Transformers 사용
# =============================================================================

# -----------------------------------------------------------------------------
# 1. 환경 변수 설정
# -----------------------------------------------------------------------------
# CUDA 메모리 할당 최적화 - 메모리 단편화 방지를 위해 최대 분할 크기를 128MB로 제한
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# -----------------------------------------------------------------------------
# 2. 모델 및 경로 설정
# -----------------------------------------------------------------------------
# 베이스 모델 경로 - 파인튜닝할 사전 훈련된 모델
BASE_MODEL_NAME="/home/project/rapa/ckpt/eeve_chatvector_OB_update"

# 체크포인트 저장 경로 - 훈련 중간 상태 저장용
CHECKPOINT_PATH="/home/project/rapa/checkpoint"

# 최종 모델 저장 경로 - 병합된 완성 모델 저장용
BASE_SAVE_PATH="/home/project/rapa/final_model"

# 훈련 데이터셋 경로 - input, output 컬럼을 포함한 데이터셋 필요
DATA_PATH="/home/project/rapa/dataset/dataset_fintuning_0603"

# -----------------------------------------------------------------------------
# 3. 훈련 하이퍼파라미터 설정
# -----------------------------------------------------------------------------
# 작업 모드: "training" (처음부터), "resume_training" (체크포인트에서 재시작)
JOB="training"

# QLoRA 사용 여부 - 메모리 효율적인 4bit 양자화 LoRA
USE_QLORA="True"

# 일반 LoRA 설정
BASE_LEARNING_RATE=5e-6      # 베이스 학습률 (보수적)
LORA_BATCH_SIZE=8            # LoRA 배치 크기

# QLoRA 설정 (더 공격적인 파라미터)
QLORA_LEARNING_RATE=1e-4     # QLoRA용 높은 학습률 (양자화로 인한 정보 손실 보상)
QLORA_BATCH_SIZE=4           # QLoRA 배치 크기 (메모리 절약)

# 공통 훈련 파라미터
NUM_EPOCHS=3                 # 전체 데이터셋 반복 횟수
LORA_RANK=16                 # LoRA 저차원 분해의 랭크 (모델 복잡도 조절)
LORA_ALPHA=32                # LoRA 스케일링 파라미터 (일반적으로 rank의 2배)

# -----------------------------------------------------------------------------
# 4. QLoRA 설정에 따른 동적 파라미터 선택
# -----------------------------------------------------------------------------
if [ "$USE_QLORA" = "True" ]; then
    CURRENT_LEARNING_RATE=$QLORA_LEARNING_RATE
    CURRENT_BATCH_SIZE=$QLORA_BATCH_SIZE
    echo "QLoRA 모드 활성화 - 4bit 양자화 사용"
else
    CURRENT_LEARNING_RATE=$BASE_LEARNING_RATE
    CURRENT_BATCH_SIZE=$LORA_BATCH_SIZE
    echo "일반 LoRA 모드 활성화"
fi

# -----------------------------------------------------------------------------
# 5. 훈련 시작 정보 출력
# -----------------------------------------------------------------------------
echo "=============================================="
echo "LLM 파인튜닝 시작"
echo "=============================================="
echo "모델: $BASE_MODEL_NAME"
echo "QLoRA 사용: $USE_QLORA"
echo "학습률: $CURRENT_LEARNING_RATE"
echo "배치 크기: $CURRENT_BATCH_SIZE"
echo "사용 GPU: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# -----------------------------------------------------------------------------
# 6. DeepSpeed를 사용한 분산 훈련 실행
# -----------------------------------------------------------------------------
# GPU 0,1,2를 사용하여 3-GPU 분산 훈련
# 각 파라미터 설명:
# --job: 훈련 모드 (training/resume_training)
# --base_model_name: 베이스 모델 경로
# --checkpoint_path: 체크포인트 저장 경로
# --save_path: 모델 저장 경로
# --data_path: 데이터셋 경로
# --learning_rate: 학습률
# --per_device_train_batch_size: 디바이스당 배치 크기
# --num_epochs: 에폭 수
# --lora_rank: LoRA 랭크
# --lora_alpha: LoRA 알파
# --use_qlora: QLoRA 사용 여부
# --torch_dtype_bf16: BFloat16 정밀도 사용 (메모리 절약)
# --use_gradient_checkpointing: 그래디언트 체크포인팅 (메모리 절약)
# --use_deepspeed: DeepSpeed 최적화 사용

CUDA_VISIBLE_DEVICES=0,1,2 deepspeed train.py \
    --job $JOB \
    --base_model_name "$BASE_MODEL_NAME" \
    --checkpoint_path $CHECKPOINT_PATH \
    --save_path $BASE_SAVE_PATH \
    --data_path $DATA_PATH \
    --learning_rate $CURRENT_LEARNING_RATE \
    --per_device_train_batch_size $CURRENT_BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --use_qlora $USE_QLORA \
    --torch_dtype_bf16 True \
    --use_gradient_checkpointing True \
    --use_deepspeed True

# -----------------------------------------------------------------------------
# 7. 훈련 완료 후 모델 병합 및 저장
# -----------------------------------------------------------------------------
echo "=============================================="
echo "LoRA 어댑터를 베이스 모델과 병합 중..."
echo "=============================================="

# LoRA 어댑터와 베이스 모델을 병합하여 완전한 모델 생성
# 각 파라미터 설명:
# --base_model_name: 베이스 모델
# --save_path: 저장 경로
# --use_qlora: QLoRA 사용 여부
# --torch_dtype_bf16: BFloat16 정밀도
# --num_epochs: 에폭 수 (체크포인트 식별용)

python model_merge.py \
    --base_model_name "$BASE_MODEL_NAME" \
    --save_path $BASE_SAVE_PATH \
    --use_qlora $USE_QLORA \
    --torch_dtype_bf16 True \
    --num_epochs $NUM_EPOCHS

# -----------------------------------------------------------------------------
# 8. 완료 메시지
# -----------------------------------------------------------------------------
echo "=============================================="
echo "파인튜닝 완료!"
echo "=============================================="
echo "모델: $BASE_MODEL_NAME"
echo "QLoRA 사용: $USE_QLORA"
echo "최종 모델 저장 위치: $BASE_SAVE_PATH"
echo "=============================================="

# =============================================================================
# 스크립트 사용법:
# 1. 데이터셋을 올바른 형식 (input, output 컬럼)으로 준비
# 2. 경로들이 실제 존재하는지 확인
# 3. GPU 메모리 상황에 따라 배치 크기 조정
# 4. chmod +x script_name.sh로 실행 권한 부여
# 5. ./script_name.sh로 실행
# =============================================================================