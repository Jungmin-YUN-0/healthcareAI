BASE_MODEL_PATH=meta-llama/Meta-Llama-3-8B
CHAT_MODEL_PATH=aaditya/Llama3-OpenBioLLM-8B
CHAT_VECTOR_PATH=ckpt_tv/Llama3-OpenBioLLM-8B-chat-vector

python extract_chat_vector.py $BASE_MODEL_PATH $CHAT_MODEL_PATH $CHAT_VECTOR_PATH