#!/bin/bash

DATA_PATH="/home/jungmin/workspace/project/rapa/data/qa_korMedMCQA_preprocessed"
RESULT_PATH="/home/jungmin/workspace/project/rapa/result/qa"

MODEL_NAME="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
CACHE_DIR="/home/project/rapa/final_model/eeve_chatvector_OB_update/1/lora_merged"

python qa_eval_sc.py \
    --data_path "$DATA_PATH" \
    --model_name "$MODEL_NAME" \
    --cache_dir "$CACHE_DIR" \
    --result_path "$RESULT_PATH" \
    --use_self_consistency True \
