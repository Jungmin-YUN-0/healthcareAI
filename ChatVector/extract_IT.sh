BASE_MODEL_PATH=beomi/Llama-3-KoEn-8B
CHAT_MODEL_PATH=beomi/Llama-3-KoEn-8B-Instruct-preview
CHAT_VECTOR_PATH=ckpt_tv/Llama3-8B-instruction-vector

python extract_chat_vector.py $BASE_MODEL_PATH $CHAT_MODEL_PATH $CHAT_VECTOR_PATH