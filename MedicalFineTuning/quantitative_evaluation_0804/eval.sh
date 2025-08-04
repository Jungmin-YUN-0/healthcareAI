#!/bin/bash

# OpenAI API Key 설정 (사용하기 전에 실제 키로 변경하세요)
export OPENAI_API_KEY=""

# RESULT_PATH 설정
RESULT_PATH="./result"

# MODEL_NAME="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
MODEL_NAMES=(
LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
/home/project/rapa/final_model_dataset_1/home/project/rapa/final_model_dataset_2/aaditya/Llama3-OpenBioLLM-8B/1/lora_merged/1/lora_merged_system
/home/project/rapa/final_model_dataset_1/home/project/rapa/final_model/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct/lora_merged/1/lora_merged_system
/home/project/rapa/final_model_dataset_2/home/project/rapa/final_model/yanolja/EEVE-Korean-10.8B-v1.0/2/lora_merged/1/lora_merged
gpt-4o
    )

# Dataset options: kormedmcqa, genmedgpt, example
DATASETS=("genmedgpt")

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
        
        python evaluation.py \
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

# 모든 모델에 대해 평가 실행
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    # OpenAI API 키 확인
    if ! check_openai_api_key "$MODEL_NAME"; then
        echo "Skipping model $MODEL_NAME due to missing API key"
        continue
    fi

    # OpenAI 모델인지 확인
    if [[ "$MODEL_NAME" == gpt-* ]]; then
        CACHE_DIR=""
        # OpenAI 모델의 경우 GPU 설정 없이 실행
        GPU_SETTING=""
        API_KEY_ARG="--openai_api_key $OPENAI_API_KEY"
    else
        if [ "$MODEL_NAME" = "/home/project/rapa/final_model_dataset_1/home/project/rapa/final_model_dataset_2/aaditya/Llama3-OpenBioLLM-8B/1/lora_merged/1/lora_merged_system" ] || \
        [ "$MODEL_NAME" = "/home/project/rapa/final_model_dataset_1/home/project/rapa/final_model/yanolja/EEVE-Korean-10.8B-v1.0/2/lora_merged/1/lora_merged_system" ] || \
        [ "$MODEL_NAME" = "/home/project/rapa/final_model_dataset_1/home/project/rapa/final_model/yanolja/EEVE-Korean-10.8B-v1.0/2/lora_merged/1/lora_merged_system" ] || \
        [ "$MODEL_NAME" = "/home/project/rapa/final_model_dataset_2/home/project/rapa/final_model/yanolja/EEVE-Korean-10.8B-v1.0/2/lora_merged/1/lora_merged" ] || \
        [ "$MODEL_NAME" = "/home/project/rapa/final_model/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct/lora_merged" ]; then
            CACHE_DIR=""
        else
            CACHE_DIR="/home/project/rapa/cache"
        fi
        # vLLM 모델의 경우 GPU 설정
        GPU_SETTING="CUDA_VISIBLE_DEVICES=1,2"
        API_KEY_ARG=""
    fi

    # Loop through datasets
    for DATASET in "${DATASETS[@]}"; do
        # 데이터셋에 따라 시스템 프롬프트 설정 결정
        if [ "$DATASET" = "kormedmcqa" ]; then
            USE_SYSTEM_PROMPT_VALUES=("False")
        elif [ "$DATASET" = "genmedgpt" ]; then
            USE_SYSTEM_PROMPT_VALUES=("True")
        fi

        # Loop through system prompt settings
        for USE_SYSTEM_PROMPT in "${USE_SYSTEM_PROMPT_VALUES[@]}"; do

            echo "Running with MODEL_NAME=$MODEL_NAME, DATASET=$DATASET, USE_SYSTEM_PROMPT=$USE_SYSTEM_PROMPT"

            # GPT-4o 모델이고 genmedgpt 데이터셋인 경우 reference 생성 스킵
            if [[ "$MODEL_NAME" == "gpt-4o" && "$DATASET" == "genmedgpt" ]]; then
                echo "Skipping GPT-4o evaluation for GenMedGPT (already generated as reference)"
                continue
            fi

            eval "$GPU_SETTING python evaluation.py \
                --model_name \"$MODEL_NAME\" \
                --cache_dir \"$CACHE_DIR\" \
                --result_path \"$RESULT_PATH\" \
                --dataset \"$DATASET\" \
                --use_system_prompt \"$USE_SYSTEM_PROMPT\" \
                --temperature 0.0 \
                --top_p 1.0 \
                $API_KEY_ARG"

            echo "Completed run with MODEL_NAME=$MODEL_NAME, DATASET=$DATASET, USE_SYSTEM_PROMPT=$USE_SYSTEM_PROMPT"
            echo "----------------------------------------"
        done
    done
done

echo "All runs completed!"
