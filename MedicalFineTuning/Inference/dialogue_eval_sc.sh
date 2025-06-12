#!/bin/bash

DATA_PATH="/home/jungmin/workspace/project/rapa/data/dialogue_koMedicalChat_preprocessed"
RESULT_PATH="/home/jungmin/workspace/project/rapa/result/dialogue"



MODEL_NAME="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"

CACHE_DIR="/home/jungmin/workspace/project/rapa/healthcareAI/ChatVector/ckpt/eeve_inst_chatvector_OB"


CUDA_VISIBLE_DEVICES=3 python dialogue_eval_sc.py \
    --data_path "$DATA_PATH" \
    --model_name "$MODEL_NAME" \
    --result_path "$RESULT_PATH" \
    --cache_dir "$CACHE_DIR"\
    --use_self_consistency True
