CP_MODEL_PATH=beomi/Llama-3-KoEn-8B
CV1_PATH=ckpt_tv/Llama3-OpenBioLLM-8B-chat-vector
CV2_PATH=ckpt_tv/Llama3-8B-instruction-vector
OUTPUT_PATH=ckpt/Llama-3-8B-KoEn-Instruction-Medical

python add_chat_vector.py $CP_MODEL_PATH "['$CV1_PATH','$CV2_PATH']" $OUTPUT_PATH \
--ratio "[0.5,0.5]"