CP_MODEL_PATH=aaditya/Llama3-OpenBioLLM-8B
CHAT_VECTOR_PATH=ckpt_tv/Llama3-KoEn-8B-korean-vector
OUTPUT_PATH=ckpt/Llama-3-8B-OpenBioLLM-Korean

python add_chat_vector.py $CP_MODEL_PATH "['$CHAT_VECTOR_PATH']" $OUTPUT_PATH \
--ratio "[1]"  # chat vector ratio