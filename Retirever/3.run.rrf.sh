#!/bin/bash

CORPUS_PATH="/path/to/dataset/msd_kdca_corpus.jsonl"
QUERY_PATH="/path/to/dataset/msd_kdca_queries.jsonl"
K=2
MODEL_PATH="nlpai-lab/KoE5"  # embedding model path (DPR) 
MAX_LENGTH=512  # embedding max_length (DPR)
EMBEDDING_PATH="/path/to/dpr_embeddings/embedding.KoE5.512.pt"

python run_retrieval.py rrf --corpus_path $CORPUS_PATH \
                            --query_path $QUERY_PATH \
                            --k $K \
                            --model_path $MODEL_PATH \
                            --embedding_path $EMBEDDING_PATH \
                            --max_length $MAX_LENGTH
