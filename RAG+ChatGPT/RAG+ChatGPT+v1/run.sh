#!/bin/bash

CORPUS_PATH="/path/to/dataset/medical_qa_corpus.jsonl"
QUERY_PATH="/path/to/dataset/medical_qa_query.jsonl"
N=5 # sampling number of query

# data preprocessing (ONLY FOR medical_qa dataset)
python preprocess_dataset.py --task query --input_path "/path/to/dataset/medical_qa/1.질문/소화기질환" --output_path $QUERY_PATH --n $N
python preprocess_dataset.py --task corpus --input_path "/path/to/dataset/medical_qa/2.답변/소화기질환" --output_path $CORPUS_PATH

K=5
MODEL_PATH="klue/bert-base"  # embedding model path (DPR)
# EMBEDDING_PATH="/path/to/embedding/embedding.bert-base.256.pt" # provide the embedding file if it exists. If not, it will be generated automatically (DPR)
MAX_LENGTH=256  # embedding max_length (DPR)
CHATGPT_MODEL="chatgpt4o"
export SYSTEM_PROMPT="당신은 의학 전문가 AI이다. 당신은 환자의 질문이나 요청에 대해 상세하고 전문적인 답변을 이해하기 쉽게 제공한다. 그러나, 당신은 의사가 아니기 때문에 직접적인 진료나 진단을 수행할 수는 없다. 이러한 경우, 당신은 환자를 의사의 진료를 받을 수 있도록 안내해야 한다."
export USER_PROMPT="[관련 정보]를 참고하여 [질문]에 답변하세요."
API_KEY='your-api-key'

# bm25
python run_retrieval.py bm25 --corpus_path $CORPUS_PATH \
                            --query_path $QUERY_PATH \
                            --k $K \
                            --chatgpt_model $CHATGPT_MODEL \
                            --system_prompt "$SYSTEM_PROMPT" \
                            --user_prompt "$USER_PROMPT" \
                            --api_key $API_KEY

# dpr
# python run_retrieval.py dpr --corpus_path $CORPUS_PATH \
#                             --query_path $QUERY_PATH \
#                             --k $K \
#                             --model_path $MODEL_PATH \
#                             --max_length $MAX_LENGTH \
#                             --chatgpt_model $CHATGPT_MODEL \
#                             --system_prompt "$SYSTEM_PROMPT" \
#                             --user_prompt "$USER_PROMPT" \
#                             --api_key $API_KEY
