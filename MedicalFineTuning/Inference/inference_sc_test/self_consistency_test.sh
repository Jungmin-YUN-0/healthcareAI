#!/bin/bash

# ==========================
# self_consistency_test.sh 실행을 위한 입력/환경 설명
# ==========================
# 1. MODEL_NAME: 사용할 HuggingFace 모델 이름 또는 로컬 경로
#    예: "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
# 2. CACHE_DIR: 모델 캐시 디렉토리 (모델이 다운로드될 경로, 사전 다운로드 권장)
#    예: "/home/project/rapa/cache"
#    로컬 경로인 경우 ""
# 3. --input_text: 테스트할 입력 문장 (단일 입력 테스트용)
# 4. --prompt: 프롬프트 템플릿 (여기서 {input}이 --input_text로 치환됨)
#    예: "[Question]: {input}\n[Options]:\n1. 약A\n2. 약B\n3. 약C\n4. 약D\n5. 약E\n\n[Answer]:"
# 5. --use_self_consistency: self-consistency(다중 샘플 생성 후 투표) 사용 여부 (옵션)
# 6. --num_samples: self-consistency 시 생성할 샘플 개수 (기본 5)
# 7. --mode: "mcqa" (객관식 1~5 답변) 또는 "openqa" (자유서술형)
#    - mcqa: 모델이 1~5 중 하나의 숫자로 답변(객관식, majority voting 적용)
#    - openqa: 모델이 자유롭게 텍스트로 답변(서술형, 확률 기반 weighted sum 적용)
#    - 프롬프트도 이에 맞게 작성 필요
# 8. --input_json: 여러 샘플을 한 번에 테스트할 경우 사용할 JSON 파일 (옵션)
#    - 파일 예시: [{"input": "질문1", "prompt": "프롬프트1"}, ...]
# 9. --output_json: 결과를 저장할 JSON 파일명 (옵션)
# ==========================

MODEL_NAME="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"   # 사용할 모델명 또는 경로
CACHE_DIR="/home/project/rapa/cache"                   # 모델 캐시 디렉토리

# 단일 입력 테스트 예시 (MCQA: 객관식, 답변은 1~5 중 하나)
# python self_consistency_test.py \
#     --model_name "$MODEL_NAME" \
#     --cache_dir "$CACHE_DIR" \
#     --input_text "당뇨병 환자의 혈당 조절에 가장 적합한 약물은 무엇입니까?" \
#     --prompt "[Question]: {input}\n[Options]:\n1. 약A\n2. 약B\n3. 약C\n4. 약D\n5. 약E\n\n[Answer]:" \
#     --use_self_consistency \
#     --num_samples 5 \
#     --mode mcqa \
#     --output_json "./self_consistency_result.json"

# open QA(서술형)로 사용하고 싶을 때는 --mode openqa와 자유 프롬프트 사용
# 예시:
python self_consistency_test.py \
    --model_name "$MODEL_NAME" \
    --cache_dir "$CACHE_DIR" \
    --input_text "당뇨병 환자의 혈당 조절에 가장 적합한 약물은 무엇입니까?" \
    --prompt "질문: {input}\n답변:" \
    --use_self_consistency \
    --num_samples 5 \
    --mode openqa \
    --output_json "./self_consistency_result_openqa.json"

# 여러 샘플(JSON) 테스트 예시
# 아래 주석을 해제하고 sample_questions.json 파일을 준비하면 여러 샘플을 한 번에 테스트할 수 있습니다.
# sample_questions.json 예시:
# [
#   {
#     "input": "고혈압 환자에게 가장 먼저 권장되는 치료법은 무엇입니까?",
#     "prompt": "[Question]: {input}\n[Options]:\n1. 식이요법\n2. 운동\n3. 약물치료\n4. 수술\n5. 기타\n\n[Answer]:"
#   },
#   {
#     "input": "감기 증상 완화에 도움이 되는 방법은?",
#     "prompt": "[Question]: {input}\n[Options]:\n1. 휴식\n2. 수분 섭취\n3. 약 복용\n4. 운동\n5. 기타\n\n[Answer]:"
#   }
# ]
# python self_consistency_test.py \
#     --model_name "$MODEL_NAME" \
#     --cache_dir "$CACHE_DIR" \
#     --input_json "sample_questions.json" \
#     --use_self_consistency \
#     --num_samples 5 \
#     --mode mcqa \
#     --output_json "self_consistency_results.json"
