EXAMPLE_PATH=./example
INPUT_PATH=./input
OUTPUT_PATH=./output
PROMPT_PATH=./example/prompt_1.txt
USE_PERMUTATION=False
GENERATION_MODEL=gpt-4o-mini
GENERATION_TEMPERATURE=1.0
GENERATION_TOPP=1.0
GENERATION_MAX_TOKENS=2048

python augment.py \
    --example_path $EXAMPLE_PATH \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_PATH \
    --prompt_path $PROMPT_PATH \
    --use_permutation $USE_PERMUTATION \
    --generation_model $GENERATION_MODEL \
    --generation_temperature $GENERATION_TEMPERATURE \
    --generation_topp $GENERATION_TOPP \
    --generation_max_tokens $GENERATION_MAX_TOKENS
