#!/bin/bash

SEARCH_TARGET_PATH="../dataset/medical_dataset_add_query.csv"
QUERY_PATH="../dataset/queries.txt"
MAX_LENGTH=512  # embedding max_length (DPR)
LLM_MODEL_NAME="yanolja/EEVE-Korean-10.8B-v1.0" # "Qwen/Qwen2.5-7B-Instruct"
EMNBEDDING_MODEL_NAME="jhgan/ko-sroberta-multitask"
GENERATION_METHOD="openllm"
K=5


python ../src/run_rag.py bm25 --search_target_path $SEARCH_TARGET_PATH \
                            --query_path $QUERY_PATH \
                            --k $K \
                            --max_length $MAX_LENGTH \
                            --llm_model_name $LLM_MODEL_NAME \
                            --embedding_model_name $EMNBEDDING_MODEL_NAME \
                            --generation_method $GENERATION_METHOD
                            # --embedding_path $EMBEDDING_PATH # 학습된 embedding이 있을 경우 해제