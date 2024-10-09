#!/bin/bash

CORPUS_PATH="/path/to/dataset/msd_kdca_corpus.jsonl"
QUERY_PATH="/path/to/dataset/msd_kdca_queries.jsonl"

K=5 
CHATGPT_MODEL="gpt-4o"
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
