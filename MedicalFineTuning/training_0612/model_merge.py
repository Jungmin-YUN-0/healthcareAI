import argparse
import os
import torch
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel

# 토크나이저 병렬 처리 경고 비활성화
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def str2bool(v):
    """
    argparse에서 bool 타입 인자를 처리하기 위한 함수
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    """
    LoRA/QLoRA 어댑터와 베이스 모델을 병합하여 최종 모델을 저장하는 스크립트
    """
    parser = argparse.ArgumentParser()
    # -------------------- 인자 정의 --------------------
    parser.add_argument("--base_model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--save_path", type=str, default="/home/project/rapa/final_model")
    parser.add_argument("--use_qlora", type=str2bool, default=True)
    parser.add_argument("--torch_dtype_bf16", type=str2bool, default=True)
    parser.add_argument("--num_epochs", type=int, default=2)

    args = parser.parse_args()

    print("🚀 모델 병합 시작")

    # -------------------- 베이스 모델 및 토크나이저 로드 --------------------
    model_path = args.base_model_name

    quantization_config = None
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.torch_dtype_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    print("📥 베이스 모델 로드...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config if args.use_qlora else None,
        torch_dtype=torch.bfloat16 if args.torch_dtype_bf16 else torch.float16,
        trust_remote_code=True,
    )

    model_short_name = args.base_model_name
    if '/home' in model_short_name:
        model_short_name = model_short_name.split('/')[-1]
    save_path = f'{args.save_path}/{model_short_name}'

    adapter_suffix = "qlora_adapters" if args.use_qlora else "lora_adapters"
    adapter_output_dir = f"{save_path}/{args.num_epochs}/{adapter_suffix}"
    merged_suffix = "qlora_merged" if args.use_qlora else "lora_merged"
    merged_output_dir = f"{save_path}/{args.num_epochs}/{merged_suffix}"

    # -------------------- 어댑터 병합 및 저장 --------------------
    print("🔗 LoRA 어댑터 로드 및 병합...")
    model = PeftModel.from_pretrained(model, adapter_output_dir)
    model = model.merge_and_unload()

    print("💾 병합된 모델 저장...")
    model.save_pretrained(merged_output_dir)

    # 토크나이저 저장
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(merged_output_dir)

    print(f"✅ 완료: {merged_output_dir}")

if __name__ == "__main__":
    main()
