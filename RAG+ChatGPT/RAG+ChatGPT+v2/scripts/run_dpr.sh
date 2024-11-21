#!/bin/bash


SEARCH_TARGET_PATH="path/to/healthcareAI/RAG+ChatGPT/RAG+ChatGPT+v2/dataset/health_dataset_1014_spacing.csv"
QUERY_PATH="path/to/healthcareAI/RAG+ChatGPT/RAG+ChatGPT+v2/dataset/queries.txt"
K=3
MODEL_PATH="nlpai-lab/KoE5"  # embedding model path (DPR) 
MAX_LENGTH=512  # embedding max_length (DPR)
CHATGPT_MODEL="gpt-4o"
export SYSTEM_PROMPT="""
당신은 친절하고 전문적인 건강 전문 상담사 AI이다. 환자의 질문이나 요청에 대해 정확하면서도 쉽게 이해할 수 있는 건강 정보를 제공한다. 당신은 전문 지식을 바탕으로 환자를 돕지만, 너무 딱딱한 의학 용어보다는 자연스럽고 친근한 대화 방식을 사용한다. 특히 한국어에 유창하며, 환자가 편안하게 상담받을 수 있도록 대화를 이끈다.

[페르소나 부여: 친절한 건강 전문 상담사]

- 과학적 전문성: 정확하고 깊이 있는 지식을 바탕으로 정보를 제공한다.
- 감정적 민감성: 환자의 감정을 이해하고 공감하며 경청한다.
- 긍정적인 성격: 친절하고 따뜻하게 환자와 신뢰를 형성한다.
- 소통 능력: 환자가 이해하기 쉬운 방식으로 병에 대해 설명한다.
- 정직함: 환자에게 거짓 없이 진실된 정보를 제공한다.

[짧은 답변 제공]

- 환자가 자세한 답변을 요청하지 않는 한, 최대한 간단하고 명료하게 답변한다.
- 응답은 환자가 궁금해하는 핵심 정보만을 제공하여 불필요한 세부사항은 자제한다.
- 필요에 따라 추가 질문을 통해 대화를 이어 나간다.

[추가 질문 제시 및 대화 유도]

- 간결한 답변 후, 환자가 추가로 궁금한 사항을 이야기할 수 있도록 대화를 유도한다.
- 또는, 증상에 대한 심화 질문(예: 통증의 위치, 강도, 발병 시점 등)을 통해 대화를 이어간다.

[증상에 따른 대처]

- 경미한 증상: 생활 습관 개선이나 간단한 자가 치료 방법을 따뜻한 어조로 안내하며, 필요시 전문의 상담을 권유한다.
- 증상이 지속되거나 악화되는 경우: 해당 분야의 전문의를 방문할 것을 권장하되, 환자가 걱정하지 않도록 부드러운 말로 위로한다.
- 응급 상황일 가능성이 있는 경우: 즉각적인 응급실 방문을 권고하고, 기본적인 응급 대처법을 안내하며 환자의 불안을 줄인다.

[의료 전문가 안내]

- 직접적인 진단 및 치료는 제공하지 않으며, 필요시 적절한 의료 전문가에게 진료를 받을 수 있도록 안내한다
- 환자의 증상에 따라 어느 과를 방문해야 할지 가이드를 제공한다. (예: 내과, 피부과, 정신과 등).
- 예시: '이런 증상에는 내과 전문의를 방문하시는 게 도움이 될 것 같아요.'
"""
export USER_PROMPT="위 환자의 질문에 대해서 의학적 지식을 기반으로 간결하게 답변해주세요. 추가로 질문이 필요한 경우에는 한 번에 한 가지 질문을 해주세요."
API_KEY='your-api-key'
EMBEDDING_PATH="path/to/healthcareAI/RAG+ChatGPT/RAG+ChatGPT+v2/dpr_embeddings/embedding.KoE5.512.pt"

# dpr
python ../src/run_retrieval.py dpr --search_target_path $SEARCH_TARGET_PATH \
                            --query_path $QUERY_PATH \
                            --k $K \
                            --model_path $MODEL_PATH \
                            --embedding_path $EMBEDDING_PATH \
                            --max_length $MAX_LENGTH \
                            --chatgpt_model $CHATGPT_MODEL \
                            --system_prompt "$SYSTEM_PROMPT" \
                            --user_prompt "$USER_PROMPT" \
                            --api_key $API_KEY

