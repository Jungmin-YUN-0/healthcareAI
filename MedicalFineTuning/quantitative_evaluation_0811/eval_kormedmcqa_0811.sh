#!/bin/bash

# ===========================================
# 평가 설정 (evaluation_0811.py 인자들 기반)
# ===========================================

# OpenAI API Key 설정
export OPENAI_API_KEY=""

# 기본 설정
RESULT_PATH="./result"
MAX_LENGTH=256
BATCH_SIZE=32

# GPU 설정
TENSOR_PARALLEL_SIZE=""  # 자동 감지하려면 비워둠
GPU_MEMORY_UTILIZATION=0.95
CLEAR_MEMORY_BEFORE_EVAL=true

# 샘플링 파라미터
DEFAULT_TEMPERATURE=0.0
DEFAULT_TOP_P=1.0

# 시스템 프롬프트 설정
SYSTEM_PROMPT_FILE=""
SYSTEM_PROMPT=""

# GPT-4o reference 생성 설정
GENERATE_GPT4O_REFERENCE=false

# 기본 모델 리스트 설정
MODEL_NAMES=(
    "trillionlabs/Tri-7B"
    "skt/A.X-4.0-Light"
    "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
)

# Fine-tuned 모델들 설정 (예시 - 필요에 따라 추가)
FINETUNED_MODELS=(
    "/home/project/rapa/final_model_0811/skt/A.X-4.0-Light/1/lora_merged_v1"
    "/home/project/rapa/final_model_0811/skt/A.X-4.0-Light/1/lora_merged_v2"
    "/home/project/rapa/final_model_0811/skt/A.X-4.0-Light/1/lora_merged_v3"
    "/home/project/rapa/final_model_0811/trillionlabs/Tri-7B/1/lora_merged_v1"
    "/home/project/rapa/final_model_0811/trillionlabs/Tri-7B/1/lora_merged_v2"
    "/home/project/rapa/final_model_0811/trillionlabs/Tri-7B/1/lora_merged_v3"
    "/home/project/rapa/final_model_0811/yanolja/EEVE-Korean-Instruct-10.8B-v1.0/1/lora_merged_v1"
    "/home/project/rapa/final_model_0811/yanolja/EEVE-Korean-Instruct-10.8B-v1.0/1/lora_merged_v2"
    "/home/project/rapa/final_model_0811/yanolja/EEVE-Korean-Instruct-10.8B-v1.0/1/lora_merged_v3"
)

# 평가 유형 리스트 (1: structured, 2: shortened, 3: shortened_50)
EVAL_TYPES=(1 2 3)

# Dataset options: kormedmcqa, genmedgpt, example
DATASETS=("kormedmcqa")

# 모델별 세부 설정
declare -A TEMPERATURE_MAP
TEMPERATURE_MAP["gpt-4o"]=0.0
TEMPERATURE_MAP["gpt-4o-mini"]=0.0
TEMPERATURE_MAP["skt/A.X-3.1"]=0.0
# 기타 모델들은 DEFAULT_TEMPERATURE 사용

declare -A TOP_P_MAP
TOP_P_MAP["gpt-4o"]=1.0
TOP_P_MAP["gpt-4o-mini"]=1.0
TOP_P_MAP["skt/A.X-3.1"]=1.0
# 기타 모델들은 DEFAULT_TOP_P 사용

declare -A CACHE_DIR_MAP
CACHE_DIR_MAP["yanolja/EEVE-Korean-Instruct-10.8B-v1.0"]="/home/project/rapa/cache"
# 로컬 모델들은 빈 문자열

# GPT 모델 사용 시 API 키 확인 함수
check_openai_api_key() {
    if [[ "$1" == gpt-* ]]; then
        if [ -z "$OPENAI_API_KEY" ]; then
            echo "Error: OPENAI_API_KEY environment variable is not set for GPT model: $1"
            echo "Please set your OpenAI API key:"
            echo "export OPENAI_API_KEY='your_api_key_here'"
            return 1
        else
            echo "OpenAI API key found for model: $1"
        fi
    fi
    return 0
}

# GPT-4o reference 답변 생성 함수
generate_gpt4o_references() {
    local dataset=$1
    
    if [ "$dataset" = "genmedgpt" ]; then
        echo "=== Generating GPT-4o reference answers for GenMedGPT dataset ==="
        
        echo "Generating GPT-4o reference with USE_SYSTEM_PROMPT=True"
        
        python evaluation_0811.py \
            --model_name "gpt-4o" \
            --cache_dir "" \
            --result_path "$RESULT_PATH" \
            --dataset "genmedgpt" \
            --use_system_prompt "True" \
            --temperature 0.0 \
            --top_p 1.0 \
            --generate_gpt4o_reference True \
            --openai_api_key "$OPENAI_API_KEY"
            
        echo "GPT-4o reference generation completed with USE_SYSTEM_PROMPT=True"
        echo "=== GPT-4o reference generation completed ==="
    fi
}

# # 각 데이터셋에 대해 GPT-4o reference 먼저 생성
# for DATASET in "${DATASETS[@]}"; do
#     generate_gpt4o_references "$DATASET"
# done

# ===========================================
# 평가 실행 함수
# ===========================================

run_evaluation() {
    local model_name="$1"
    local dataset="$2"
    local eval_type="$3"
    local use_system_prompt="$4"
    local cache_dir="$5"
    local gpu_setting="$6"
    local api_key_arg="$7"
    
    # 모델별 파라미터 설정
    local temperature=${TEMPERATURE_MAP[$model_name]:-$DEFAULT_TEMPERATURE}
    local top_p=${TOP_P_MAP[$model_name]:-$DEFAULT_TOP_P}
    
    echo "========================================"
    echo "Starting evaluation:"
    echo "  Model: $model_name"
    echo "  Dataset: $dataset"
    echo "  Type: $eval_type"
    echo "  Use system prompt: $use_system_prompt"
    echo "  Temperature: $temperature"
    echo "  Top-p: $top_p"
    echo "  Cache dir: $cache_dir"
    echo "  Max length: $MAX_LENGTH"
    echo "  GPU utilization: $GPU_MEMORY_UTILIZATION"
    echo "----------------------------------------"

    # evaluation_0811.py 실행
    local cmd="$gpu_setting python evaluation_0811.py"
    cmd="$cmd --model_name \"$model_name\""
    cmd="$cmd --cache_dir \"$cache_dir\""
    cmd="$cmd --result_path \"$RESULT_PATH\""
    cmd="$cmd --dataset \"$dataset\""
    cmd="$cmd --use_system_prompt \"$use_system_prompt\""
    cmd="$cmd --temperature \"$temperature\""
    cmd="$cmd --top_p \"$top_p\""
    cmd="$cmd --max_length \"$MAX_LENGTH\""
    cmd="$cmd --gpu_memory_utilization \"$GPU_MEMORY_UTILIZATION\""
    # cmd="$cmd --clear_memory_before_eval \"$CLEAR_MEMORY_BEFORE_EVAL\""
    
    # Type 설정 (지정된 경우에만)
    if [ "$eval_type" != "" ]; then
        cmd="$cmd --type \"$eval_type\""
    fi
    
    # 텐서 병렬 크기 설정 (지정된 경우에만)
    if [ "$TENSOR_PARALLEL_SIZE" != "" ]; then
        cmd="$cmd --tensor_parallel_size \"$TENSOR_PARALLEL_SIZE\""
    fi
    
    # 시스템 프롬프트 파일 설정
    if [ "$SYSTEM_PROMPT_FILE" != "" ]; then
        cmd="$cmd --system_prompt_file \"$SYSTEM_PROMPT_FILE\""
    fi
    
    if [ "$SYSTEM_PROMPT" != "" ]; then
        cmd="$cmd --system_prompt \"$SYSTEM_PROMPT\""
    fi
    
    # OpenAI 설정 추가
    if [ "$api_key_arg" != "" ]; then
        cmd="$cmd $api_key_arg"
    fi
    
    # GPT-4o reference 생성 설정
    if [ "$GENERATE_GPT4O_REFERENCE" = "true" ]; then
        cmd="$cmd --generate_gpt4o_reference true"
    fi
    
    echo "Executing: $cmd"
    echo ""
    
    eval $cmd
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Evaluation completed successfully"
    else
        echo "❌ Evaluation failed with exit code: $exit_code"
    fi
    
    echo "========================================"
    echo ""
}

# ===========================================
# 기본 모델들에 대해 모든 타입으로 평가 실행
# ===========================================

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    # OpenAI API 키 확인
    if ! check_openai_api_key "$MODEL_NAME"; then
        echo "Skipping model $MODEL_NAME due to missing API key"
        continue
    fi

    # 캐시 디렉토리 설정
    if [[ "$MODEL_NAME" == gpt-* ]]; then
        CACHE_DIR="/home/project/rapa/cache"
        GPU_SETTING=""
        API_KEY_ARG="--openai_api_key $OPENAI_API_KEY"
    else
        if [[ "$MODEL_NAME" == /home/* ]] || [[ "$MODEL_NAME" == /home/project/* ]]; then
            CACHE_DIR=""
        else
            CACHE_DIR=${CACHE_DIR_MAP[$MODEL_NAME]:-"/home/project/rapa/cache"}
        fi
        GPU_SETTING="CUDA_VISIBLE_DEVICES=0"
        API_KEY_ARG=""
    fi

    # Loop through datasets
    for DATASET in "${DATASETS[@]}"; do
        # 데이터셋에 따라 시스템 프롬프트 설정 결정
        if [ "$DATASET" = "kormedmcqa" ]; then
            USE_SYSTEM_PROMPT_VALUES=("False")
        elif [ "$DATASET" = "genmedgpt" ]; then
            USE_SYSTEM_PROMPT_VALUES=("True")
        else
            USE_SYSTEM_PROMPT_VALUES=("False")
        fi

        # Loop through system prompt settings
        for USE_SYSTEM_PROMPT in "${USE_SYSTEM_PROMPT_VALUES[@]}"; do
            # GPT-4o 모델이고 genmedgpt 데이터셋인 경우 reference 생성 스킵
            if [[ "$MODEL_NAME" == "gpt-4o" && "$DATASET" == "genmedgpt" ]]; then
                echo "Skipping GPT-4o evaluation for GenMedGPT (already generated as reference)"
                continue
            fi
            
            # 각 평가 유형에 대해 실행
            for EVAL_TYPE in "${EVAL_TYPES[@]}"; do
                run_evaluation "$MODEL_NAME" "$DATASET" "$EVAL_TYPE" "$USE_SYSTEM_PROMPT" "$CACHE_DIR" "$GPU_SETTING" "$API_KEY_ARG"
            done
        done
    done
done

# ===========================================
# Fine-tuned 모델들에 대해 해당하는 타입으로만 평가 실행
# ===========================================

for MODEL_NAME in "${FINETUNED_MODELS[@]}"; do
    # 캐시 디렉토리 설정 (로컬 모델)
    CACHE_DIR=""
    GPU_SETTING="CUDA_VISIBLE_DEVICES=0"
    API_KEY_ARG=""

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

    # Loop through datasets
    for DATASET in "${DATASETS[@]}"; do
        # 데이터셋에 따라 시스템 프롬프트 설정 결정
        if [ "$DATASET" = "kormedmcqa" ]; then
            USE_SYSTEM_PROMPT_VALUES=("False")
        elif [ "$DATASET" = "genmedgpt" ]; then
            USE_SYSTEM_PROMPT_VALUES=("True")
        else
            USE_SYSTEM_PROMPT_VALUES=("False")
        fi

        # Loop through system prompt settings
        for USE_SYSTEM_PROMPT in "${USE_SYSTEM_PROMPT_VALUES[@]}"; do
            run_evaluation "$MODEL_NAME" "$DATASET" "$EVAL_TYPE" "$USE_SYSTEM_PROMPT" "$CACHE_DIR" "$GPU_SETTING" "$API_KEY_ARG"
        done
    done
done

echo "All runs completed!"
echo ""
echo "EVALUATION SUMMARY:"
echo "- Base models evaluated: ${#MODEL_NAMES[@]} (all types: ${EVAL_TYPES[@]})"
echo "- Fine-tuned models evaluated: ${#FINETUNED_MODELS[@]} (specific types)"
echo "- Datasets: ${DATASETS[@]}"
echo "- Results directory: $RESULT_PATH"
