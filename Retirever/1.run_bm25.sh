#!/bin/bash

CORPUS_PATH="/path/to/dataset/msd_kdca_corpus.jsonl"
QUERY_PATH="/path/to/dataset/msd_kdca_queries.jsonl"
K=2

python run_retrieval.py bm25 --corpus_path $CORPUS_PATH \
                            --query_path $QUERY_PATH \
                            --k $K \
