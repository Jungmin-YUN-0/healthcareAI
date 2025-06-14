CP_MODEL_PATH=beomi/Llama-3-KoEn-8B
CHAT_VECTOR_PATH=ckpt_tv/Llama3-OpenBioLLM-8B-chat-vector
OUTPUT_PATH=ckpt/Llama-3-8B-KoEn-Medical
CACHE_DIR=[cache directory]

python add_chat_vector.py $CP_MODEL_PATH "['$CHAT_VECTOR_PATH']" $OUTPUT_PATH \
--cache_dir "$CACHE_DIR" \
--ratio "[1]"  # chat vector ratio
