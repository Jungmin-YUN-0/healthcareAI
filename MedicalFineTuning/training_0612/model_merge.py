import argparse
import torch
import os
from transformers import (
    BitsAndBytesConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer
)
from peft import PeftModel

# 토크나이저 병렬 처리 경고 비활성화
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--base_model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--save_path", type=str, default="/home/project/rapa/final_model")
    parser.add_argument("--use_qlora", type=str2bool, default=True)
    parser.add_argument("--torch_dtype_bf16", type=str2bool, default=True)
    parser.add_argument("--num_epochs", type=int, default=2)
    
    args = parser.parse_args()
    
    print("🚀 모델 병합 시작")
    
    # Load model and tokenizer based on model name
    model_path = args.base_model_name

    # Configure quantization if using QLoRA
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
    
    print("🔗 LoRA 어댑터 로드 및 병합...")
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_output_dir)
    model = model.merge_and_unload() 
    
    print("💾 병합된 모델 저장...")
    
    model.save_pretrained(merged_output_dir)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(merged_output_dir)
    
    print(f"✅ 완료: {merged_output_dir}")
    
if __name__ == "__main__":
    main()
