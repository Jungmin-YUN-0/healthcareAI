#!/bin/bash

DATA_PATH="/home/jungmin/workspace/project/rapa/data/dialogue_koMedicalChat_preprocessed"
RESULT_PATH="/home/jungmin/workspace/project/rapa/result/dialogue"

MODEL_NAME="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
CACHE_DIR="/home/project/rapa/final_model/eeve_chatvector_OB_update/1/lora_merged"

python dialogue_eval.py \
    --data_path "$DATA_PATH" \
    --model_name "$MODEL_NAME" \
    --result_path "$RESULT_PATH" \
    --cache_dir "$CACHE_DIR"
