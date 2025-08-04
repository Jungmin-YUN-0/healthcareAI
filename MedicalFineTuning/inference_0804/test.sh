DATA_PATH="/home/project/rapa/dataset/dataset_fintuning_0603"
RESULT_PATH="./result/0731"
# MODEL_NAME="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
MODEL_NAMES=(
    "aaditya/Llama3-OpenBioLLM-8B"
    "/home/project/rapa/final_model_dataset_2/aaditya/Llama3-OpenBioLLM-8B/1/lora_merged"
    "/home/project/rapa/final_model_dataset_1/home/project/rapa/final_model_dataset_2/aaditya/Llama3-OpenBioLLM-8B/1/lora_merged/1/lora_merged_system")

    # "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    # "/home/project/rapa/final_model/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct/lora_merged"
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    if [ "$MODEL_NAME" = "/home/project/rapa/final_model/yanolja/EEVE-Korean-10.8B-v1.0/2/lora_merged_system" ] || \
    [ "$MODEL_NAME" = "/home/project/rapa/final_model/yanolja/EEVE-Korean-10.8B-v1.0/2/lora_merged" ] || \
    [ "$MODEL_NAME" = "/home/project/rapa/final_model_dataset_1/home/project/rapa/final_model/yanolja/EEVE-Korean-10.8B-v1.0/2/lora_merged/1/lora_merged_system" ] || \
    [ "$MODEL_NAME" = "/home/project/rapa/final_model_dataset_2/home/project/rapa/final_model/yanolja/EEVE-Korean-10.8B-v1.0/2/lora_merged/1/lora_merged" ] || \
    [ "$MODEL_NAME" = "/home/project/rapa/final_model/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct/lora_merged" ]; then
        CACHE_DIR=""
    else
        CACHE_DIR="/home/project/rapa/cache"
    fi

    # Loop through both system prompt settings
    for USE_SYSTEM_PROMPT in "False" "True"; do
        echo "Running with MODEL_NAME=$MODEL_NAME, USE_SYSTEM_PROMPT=$USE_SYSTEM_PROMPT"
        if [ "$USE_SYSTEM_PROMPT" = "True" ]; then
            MAX_LENGTH=4096
        else
            MAX_LENGTH=1024
        fi

        CUDA_VISIBLE_DEVICES=0,1 python test.py \
            --model_name "$MODEL_NAME" \
            --cache_dir "$CACHE_DIR" \
            --result_path "$RESULT_PATH" \
            --use_system_prompt "$USE_SYSTEM_PROMPT" \
            --max_length "$MAX_LENGTH" \
            --temperature 0.0 \
            # --top_p 1.0 \
            # --data_path "$DATA_PATH"


            # --num_samples 100
        
        echo "Completed run with MODEL_NAME=$MODEL_NAME, USE_SYSTEM_PROMPT=$USE_SYSTEM_PROMPT"
        echo "----------------------------------------"
    done
done

echo "All runs completed!"
