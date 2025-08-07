# 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 모델 리스트 설정
BASE_MODEL_NAMES=("")
# QLoRA 설정 리스트
USE_QLORA_OPTIONS=("False")
# "False"

# 시스템 프롬프트 설정 리스트
USE_SYSTEM_PROMPT_OPTIONS=("False")
# "True"

# 공통 경로 설정
CHECKPOINT_PATH="/home/project/rapa/checkpoint_dataset_2"
CACHE_PATH="/home/project/rapa/cache"
BASE_SAVE_PATH="/home/project/rapa/final_model_dataset_2"
DATA_PATH="/home/project/rapa/dataset/dataset_finetuning_250723_v2"

# 학습 하이퍼파라미터
BASE_LEARNING_RATE=5e-6
QLORA_LEARNING_RATE=1e-4  # QLoRA용 더 큰 학습ㅅ률
QLORA_BATCH_SIZE=4
LORA_BATCH_SIZE=8
NUM_EPOCHS=1
LORA_RANK=16
LORA_ALPHA=32

# 에폭 리스트 설정
EPOCH_LIST=(1)

# 각 모델에 대해 학습 실행
for BASE_MODEL_NAME in "${BASE_MODEL_NAMES[@]}"; do
    for USE_QLORA in "${USE_QLORA_OPTIONS[@]}"; do
        for USE_SYSTEM_PROMPT in "${USE_SYSTEM_PROMPT_OPTIONS[@]}"; do
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

                # 시스템 프롬프트 사용 여부에 따른 max_length 및 배치 설정
                if [ "$USE_SYSTEM_PROMPT" = "True" ]; then
                    MAX_LENGTH=4096
                    # 원래 배치 크기를 저장하고 절반으로 줄임
                    ORIGINAL_BATCH_SIZE=$CURRENT_BATCH_SIZE
                    CURRENT_BATCH_SIZE=2
                    # gradient accumulation으로 원래 배치 크기 효과 유지
                    GRADIENT_ACCUMULATION_STEPS=$((ORIGINAL_BATCH_SIZE / CURRENT_BATCH_SIZE))
                    # 최소 1로 설정
                    if [ "$GRADIENT_ACCUMULATION_STEPS" -lt 1 ]; then
                        GRADIENT_ACCUMULATION_STEPS=1
                    fi
                else
                    MAX_LENGTH=512
                    GRADIENT_ACCUMULATION_STEPS=1
                fi

                echo "Starting training for model: $BASE_MODEL_NAME with QLoRA: $USE_QLORA, System Prompt: $USE_SYSTEM_PROMPT, epoch: $NUM_EPOCHS, job: $JOB_TYPE"
                echo "Using learning rate: $CURRENT_LEARNING_RATE"
                echo "Using batch size: $CURRENT_BATCH_SIZE"
                echo "Using gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
                echo "Using max length: $MAX_LENGTH"
                echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

                # DeepSpeed에서 특정 GPU 지정하여 실행
            CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port 29501 train_dataset_2.py \
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
                    --use_system_prompt $USE_SYSTEM_PROMPT \
                    --torch_dtype_bf16 True \
                    --use_gradient_checkpointing True \
                    --use_deepspeed True

                python model_merge.py \
                    --base_model_name "$BASE_MODEL_NAME" \
                    --cache_path $CACHE_PATH \
                    --save_path $BASE_SAVE_PATH \
                    --use_qlora $USE_QLORA \
                    --use_system_prompt $USE_SYSTEM_PROMPT \
                    --torch_dtype_bf16 True \
                    --num_epochs $NUM_EPOCHS

                echo "Completed training for model: $BASE_MODEL_NAME with QLoRA: $USE_QLORA, System Prompt: $USE_SYSTEM_PROMPT, epoch: $NUM_EPOCHS"
                echo "Model saved to: $SAVE_PATH/$BASE_MODEL_NAME"
                echo "----------------------------------------"
            done
        done
    done
done

echo "All training completed!"
