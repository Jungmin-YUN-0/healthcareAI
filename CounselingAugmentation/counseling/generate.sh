#!/bin/bash

# ====================================================
# SCRIPT CONFIGURATION
# ====================================================
# 작업 실행 모드 선택 ('summary' 또는 'dialogue')

#aug_trg='summary'
aug_trg='dialogue'

# 작업별 공통 인자 정의
ARGS="--example_path "./data_sample" \
    --generation_model "gpt-4o" \
    --generation_top_p 0.95 \
    --shot_mode 1"  
# 0: zero-shot, 1: one-shot, 2: few-shot (permutation)


# ====================================================
# TASK DEFINITIONS
# ====================================================
# 'summary' 생성 작업 실행
if [ "$aug_trg" = "summary" ]; then
    PYFILE=augment_summary.py
    ARGS="$ARGS \
    --output_path "./output_summary" \
    --prompt_path "./prompt_summary.txt" \
    --generation_temperature 0.95 \
    --generation_max_tokens 4096"
   
# 'dialogue' 생성 작업 실행    
elif [ "$aug_trg" = "dialogue" ]; then
    PYFILE=augment_dialogue.py
    ARGS="$ARGS \
    --input_path "./output_summary" \
    --output_path "./output_dialogue" \
    --prompt_path "./prompt_dialogue.txt" \
    --generation_temperature 0.8 \
    --generation_max_tokens 16000"

else
    echo "Unknown aug_trg: $aug_trg"
    exit 1
fi

# ====================================================
# MAIN EXECUTION LOGIC
# ====================================================
# 스크립트 실행
echo "-------------------------------------"
echo "Running Command:"
echo "python $PYFILE $ARGS"
echo "-------------------------------------"

eval python $PYFILE $ARGS