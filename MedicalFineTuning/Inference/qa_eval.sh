#!/bin/bash

DATA_PATH="/home/jungmin/workspace/project/rapa/data/qa_korMedMCQA_preprocessed"
RESULT_PATH="/home/jungmin/workspace/project/rapa/result/qa"

#CACHE_DIR="/nas_homes/projects/rapa/Chatvector"
#MODEL_NAME="yanolja/EEVE-Korean-10.8B-v1.0"
MODEL_NAME="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
#MODEL_NAME="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

#CACHE_DIR="/home/jungmin/workspace/project/rapa/healthcareAI/ChatVector/ckpt/eeve_chatvector_OB"
#CACHE_DIR="/home/jungmin/workspace/project/rapa/healthcareAI/ChatVector/ckpt/eeve_chatvector_OB"
#CACHE_DIR="/home/jungmin/workspace/project/rapa/healthcareAI/ChatVector/ckpt/eeve_inst_chatvector_OB"
#CACHE_DIR="/home/project/rapa/final_model/eeve_inst_chatvector_OB/lora_merged"
CACHE_DIR="/home/jungmin/workspace/project/rapa/healthcareAI/ChatVector/ckpt/eeve_chatvector_OB_update"
CACHE_DIR="/home/project/rapa/final_model/eeve_chatvector_OB_update/1/lora_merged"
#CACHE_DIR="/home/project/rapa/final_model/EEVE+ChatVector/qlora_merged"
#CACHE_DIR="/home/project/rapa/final_model/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct/qlora_merged"

#MODEL_NAME="google/medgemma-4b-it"

CUDA_VISIBLE_DEVICES=3 python qa_eval.py \
    --data_path "$DATA_PATH" \
    --model_name "$MODEL_NAME" \
    --cache_dir "$CACHE_DIR" \
    --result_path "$RESULT_PATH" \

