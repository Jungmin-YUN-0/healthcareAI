# 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 모델 리스트 설정
BASE_MODEL_NAMES=("trillionlabs/Tri-7B" "skt/A.X-4.0-Light")
# "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct" "EEVE+ChatVector" "eeve_inst_chatvector_OB" "eeve_chatvector_OB_update"
# QLoRA 설정 리스트
USE_QLORA_OPTIONS=("False")
# "False"

# 학습 유형 리스트 (1: structured, 2: shortened, 3: shortened_50)
TYPE_OPTIONS=(1 2 3)

# 공통 경로 설정
CHECKPOINT_PATH="./checkpoint"
CACHE_PATH="./cache"
BASE_SAVE_PATH="./final_model"
DATA_PATH="/home/project/rapa/dataset/finetuning_dataset_250811"

# 학습 하이퍼파라미터
BASE_LEARNING_RATE=5e-6
QLORA_LEARNING_RATE=1e-4  # QLoRA용 더 큰 학습률
QLORA_BATCH_SIZE=4
LORA_BATCH_SIZE=4
NUM_EPOCHS=1
LORA_RANK=16
LORA_ALPHA=32

# 에폭 리스트 설정
EPOCH_LIST=(1)

# 각 모델에 대해 학습 실행
for BASE_MODEL_NAME in "${BASE_MODEL_NAMES[@]}"; do
    for USE_QLORA in "${USE_QLORA_OPTIONS[@]}"; do
        for TYPE in "${TYPE_OPTIONS[@]}"; do
            for NUM_EPOCHS in "${EPOCH_LIST[@]}"; do
                if [ "$NUM_EPOCHS" -eq 1 ]; then
                    JOB_TYPE="training"
                else
                    JOB_TYPE="resume_training"
                fi

                # QLoRA 설정에 따른 학습률 결정
                if [ "$USE_QLORA" = "True" ]; then
                    CURRENT_LEARNING_RATE=$QLORA_LEARNING_RATE
                    CURRENT_BATCH_SIZE=$QLORA_BATCH_SIZE
                else
                    CURRENT_LEARNING_RATE=$BASE_LEARNING_RATE
                    CURRENT_BATCH_SIZE=$LORA_BATCH_SIZE
                fi

                # 유형별로 max_length 및 배치 설정 조정
                MAX_LENGTH=4096
                # gradient accumulation steps를 1로 고정
                GRADIENT_ACCUMULATION_STEPS=2

                echo "Starting training for model: $BASE_MODEL_NAME with QLoRA: $USE_QLORA, Type: $TYPE, epoch: $NUM_EPOCHS, job: $JOB_TYPE"
                echo "Using learning rate: $CURRENT_LEARNING_RATE"
                echo "Using batch size: $CURRENT_BATCH_SIZE"
                echo "Using gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
                echo "Using max length: $MAX_LENGTH"
                echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

                # DeepSpeed에서 특정 GPU 지정하여 실행
                deepspeed train_0811.py \
                    --job "$JOB_TYPE" \
                    --base_model_name "$BASE_MODEL_NAME" \
                    --checkpoint_path $CHECKPOINT_PATH \
                    --cache_path $CACHE_PATH \
                    --save_path $BASE_SAVE_PATH \
                    --data_path $DATA_PATH \
                    --learning_rate $CURRENT_LEARNING_RATE \
                    --per_device_train_batch_size $CURRENT_BATCH_SIZE \
                    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
                    --num_epochs $NUM_EPOCHS \
                    --max_length $MAX_LENGTH \
                    --lora_rank $LORA_RANK \
                    --lora_alpha $LORA_ALPHA \
                    --use_qlora $USE_QLORA \
                    --type $TYPE \
                    --torch_dtype_bf16 True \
                    --use_gradient_checkpointing True \
                    --use_deepspeed True

                python model_merge_0811.py \
                    --base_model_name "$BASE_MODEL_NAME" \
                    --cache_path $CACHE_PATH \
                    --save_path $BASE_SAVE_PATH \
                    --use_qlora $USE_QLORA \
                    --type $TYPE \
                    --torch_dtype_bf16 True \
                    --num_epochs $NUM_EPOCHS

                echo "Completed training for model: $BASE_MODEL_NAME with QLoRA: $USE_QLORA, Type: $TYPE, epoch: $NUM_EPOCHS"
                echo "Model saved to: $SAVE_PATH/$BASE_MODEL_NAME"
                echo "----------------------------------------"
            done
        done
    done
done

echo "All training completed!"
