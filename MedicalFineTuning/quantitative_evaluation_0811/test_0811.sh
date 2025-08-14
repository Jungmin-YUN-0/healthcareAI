#!/bin/bash

# 데이터셋 경로 설정
DATA_PATH="/home/project/rapa/dataset/finetuning_dataset_250811"
# DATA_PATH=''
RESULT_PATH="./result"

# 모델 리스트 설정
MODEL_NAMES=(
    "trillionlabs/Tri-7B"
    "skt/A.X-4.0-Light"
)

# Fine-tuned 모델들과 해당 type 매핑
FINETUNED_MODELS=(
    "/home/project/rapa/final_model/skt/A.X-4.0-Light/1/lora_merged_v1"
    "/home/project/rapa/final_model/skt/A.X-4.0-Light/1/lora_merged_v2"
    "/home/project/rapa/final_model/skt/A.X-4.0-Light/1/lora_merged_v3"
    "/home/project/rapa/final_model/trillionlabs/Tri-7B/1/lora_merged_v1"
    "/home/project/rapa/final_model/trillionlabs/Tri-7B/1/lora_merged_v2"
    "/home/project/rapa/final_model/trillionlabs/Tri-7B/1/lora_merged_v3"
)

# v1->type1, v2->type2, v3->type3 매핑
declare -A MODEL_TYPE_MAP
MODEL_TYPE_MAP["/home/project/rapa/final_model/yanolja/EEVE-Korean-Instruct-10.8B-v1.0/1/lora_merged_v1"]=1
MODEL_TYPE_MAP["/home/project/rapa/final_model/yanolja/EEVE-Korean-Instruct-10.8B-v1.0/1/lora_merged_v2"]=2
MODEL_TYPE_MAP["/home/project/rapa/final_model/yanolja/EEVE-Korean-Instruct-10.8B-v1.0/1/lora_merged_v3"]=3

# 평가 유형 리스트 (1: structured, 2: shortened, 3: shortened_50)
EVAL_TYPES=(1 2 3)

# Temperature 설정 (모델별)
declare -A TEMPERATURE_MAP
TEMPERATURE_MAP["dnotitia/DNA-2.0-14B"]=0.7
# 기타 모델들은 기본값 0.0 사용

# Top-p 설정 (모델별)
declare -A TOP_P_MAP  
TOP_P_MAP["dnotitia/DNA-2.0-14B"]=0.8
# 기타 모델들은 기본값 1.0 사용

# 기본 모델들에 대해 모든 타입으로 평가 실행
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    # 캐시 디렉토리 설정 (로컬 모델인지 허깅페이스 모델인지 구분)
    if [[ "$MODEL_NAME" == /home/* ]] || [[ "$MODEL_NAME" == /home/project/* ]]; then
        CACHE_DIR=""
    else
        CACHE_DIR="/home/project/rapa/cache"
    fi

    # 모델별 temperature, top_p 설정
    TEMPERATURE=${TEMPERATURE_MAP[$MODEL_NAME]:-0.0}
    TOP_P=${TOP_P_MAP[$MODEL_NAME]:-1.0}

    # 각 평가 유형에 대해 실행
    for EVAL_TYPE in "${EVAL_TYPES[@]}"; do
        echo "Starting evaluation for base model: $MODEL_NAME with type: $EVAL_TYPE"
        echo "Using temperature: $TEMPERATURE, top_p: $TOP_P"
        echo "Cache directory: $CACHE_DIR"
        echo "Data path: $DATA_PATH"
        echo "----------------------------------------"

        # GPU 0에서 평가 실행
        CUDA_VISIBLE_DEVICES=0 python test_0811.py \
            --model_name "$MODEL_NAME" \
            --cache_dir "$CACHE_DIR" \
            --result_path "$RESULT_PATH" \
            --data_path "$DATA_PATH" \
            --type "$EVAL_TYPE" \
            --temperature "$TEMPERATURE" \
            --top_p "$TOP_P" \
            --max_length 512

        echo "Completed evaluation for base model: $MODEL_NAME with type: $EVAL_TYPE"
        echo "=========================================="
    done
done

# Fine-tuned 모델들에 대해 해당하는 타입으로만 평가 실행
for MODEL_NAME in "${FINETUNED_MODELS[@]}"; do
    # 캐시 디렉토리 설정 (로컬 모델)
    CACHE_DIR=""

    # 모델별 temperature, top_p 설정
    TEMPERATURE=${TEMPERATURE_MAP[$MODEL_NAME]:-0.0}
    TOP_P=${TOP_P_MAP[$MODEL_NAME]:-1.0}

    # 해당 모델의 매핑된 타입 가져오기 (패턴 매칭 사용)
    if [[ "$MODEL_NAME" == *"lora_merged_v1" ]]; then
        EVAL_TYPE=1
    elif [[ "$MODEL_NAME" == *"lora_merged_v2" ]]; then
        EVAL_TYPE=2
    elif [[ "$MODEL_NAME" == *"lora_merged_v3" ]]; then
        EVAL_TYPE=3
    else
        echo "ERROR: No type mapping found for model: $MODEL_NAME"
        continue
    fi

    echo "Starting evaluation for fine-tuned model: $MODEL_NAME with type: $EVAL_TYPE"
    echo "Using temperature: $TEMPERATURE, top_p: $TOP_P"
    echo "Cache directory: $CACHE_DIR"
    echo "Data path: $DATA_PATH"
    echo "----------------------------------------"

    # GPU 0에서 평가 실행
    CUDA_VISIBLE_DEVICES=0 python test_0811.py \
        --model_name "$MODEL_NAME" \
        --cache_dir "$CACHE_DIR" \
        --result_path "$RESULT_PATH" \
        --data_path "$DATA_PATH" \
        --type "$EVAL_TYPE" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_length 512

    echo "Completed evaluation for fine-tuned model: $MODEL_NAME with type: $EVAL_TYPE"
    echo "=========================================="
done

echo "All evaluations completed!"
echo "Results saved in: $RESULT_PATH"

# 결과 요약 출력
echo ""
echo "EVALUATION SUMMARY:"
echo "- Base models evaluated: ${#MODEL_NAMES[@]} (all types)"
echo "- Fine-tuned models evaluated: ${#FINETUNED_MODELS[@]} (specific types)"
echo "- Evaluation types: ${#EVAL_TYPES[@]} (structured, shortened, shortened_50)"  
echo "- Total runs: $((${#MODEL_NAMES[@]} * ${#EVAL_TYPES[@]} + ${#FINETUNED_MODELS[@]}))"
echo "- Dataset: $DATA_PATH"
echo "- Results directory: $RESULT_PATH"
echo ""
echo "Model-Type Mapping:"
echo "- lora_merged_v1 -> type 1 (structured)"
echo "- lora_merged_v2 -> type 2 (shortened)"
echo "- lora_merged_v3 -> type 3 (shortened_50)"