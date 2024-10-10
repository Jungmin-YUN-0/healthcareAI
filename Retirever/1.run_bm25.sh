#!/bin/bash

CORPUS_PATH="/home/jinhee/IIPL_PROJECT/rapa/RAG/dataset/msd_kdca_corpus.jsonl"
QUERY_PATH="/home/jinhee/IIPL_PROJECT/rapa/RAG/dataset/msd_kdca_queries.jsonl"
K=2

python run_retrieval.py bm25 --corpus_path $CORPUS_PATH \
                            --query_path $QUERY_PATH \
                            --k $K \