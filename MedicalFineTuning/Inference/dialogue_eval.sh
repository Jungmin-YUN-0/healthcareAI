#!/bin/bash

DATA_PATH="/home/jungmin/workspace/project/rapa/data/dialogue_koMedicalChat_preprocessed"
RESULT_PATH="/home/jungmin/workspace/project/rapa/result/dialogue"

# MODEL_NAME="google/medgemma-4b-it"
# CACHE_DIR=""


#MODEL_NAME="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
#MODEL_NAME="yanolja/EEVE-Korean-10.8B-v1.0"
MODEL_NAME="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"

# MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
#MODEL_NAME="Qwen/Qwen3-8B"
#MODEL_NAME="Qwen/Qwen3-14B"
#CACHE_DIR="/nas_homes/seunguk/hf_llm"
#MODEL_NAME="yanolja/EEVE-Korean-10.8B-v1.0"
#CACHE_DIR="/home/jungmin/workspace/project/rapa/healthcareAI/ChatVector/ckpt/eeve_chatvector_OB"
#CACHE_DIR="/home/jungmin/workspace/project/rapa/healthcareAI/ChatVector/ckpt/eeve_chatvector_OB"

#CACHE_DIR="/home/project/rapa/final_model/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct/qlora_merged"
#CACHE_DIR="/home/project/rapa/final_model/EEVE+ChatVector/qlora_merged"
#CACHE_DIR="/home/jungmin/workspace/project/rapa/healthcareAI/ChatVector/ckpt/eeve_inst_chatvector_OB"
#CACHE_DIR="/home/project/rapa/final_model/eeve_inst_chatvector_OB/qlora_merged"
#CACHE_DIR="/home/jungmin/workspace/project/rapa/healthcareAI/ChatVector/ckpt/eeve_chatvector_OB_update"
CACHE_DIR="/home/project/rapa/final_model/eeve_chatvector_OB_update/1/lora_merged"
#CACHE_DIR=""

CUDA_VISIBLE_DEVICES=3 python dialogue_eval.py \
    --data_path "$DATA_PATH" \
    --model_name "$MODEL_NAME" \
    --result_path "$RESULT_PATH" \
    --cache_dir "$CACHE_DIR"
