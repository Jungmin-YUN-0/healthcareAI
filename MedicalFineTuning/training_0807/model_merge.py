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
    parser.add_argument("--cache_path", type=str)
    parser.add_argument("--use_qlora", type=str2bool, default=True)
    parser.add_argument("--torch_dtype_bf16", type=str2bool, default=True)
    parser.add_argument("--num_epochs", type=int, default=2)
    
    # System prompt arguments
    parser.add_argument("--use_system_prompt", type=str2bool, default=False,
                       help="Whether system prompt was used during training")
    
    args = parser.parse_args()
    
    print("🚀 모델 병합 시작")
    
    # Load model and tokenizer based on model name
    if args.base_model_name == "EEVE+ChatVector":
        model_name = 'yanolja/EEVE-Korean-10.8B-v1.0'
        model_path = "/home/project/rapa/ckpt/eeve_chatvector_IT_OB"
    elif args.base_model_name == "eeve_inst_chatvector_OB":
        model_name = '/home/project/rapa/ckpt/eeve_inst_chatvector_OB'
        model_path = "/home/project/rapa/ckpt/eeve_inst_chatvector_OB"
    
    elif args.base_model_name == "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct":
        model_name = args.base_model_name
        model_path = args.base_model_name
    elif args.base_model_name == 'eeve_chatvector_OB_update':
        model_name = '/home/project/rapa/ckpt/eeve_chatvector_OB_update'
        model_path = '/home/project/rapa/ckpt/eeve_chatvector_OB_update'
    else:
        model_name = args.base_model_name
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
        # cache_dir=args.cache_path
    )
    
    save_path = f'{args.save_path}/{args.base_model_name}'
    
    system_prompt_suffix = "_system" if args.use_system_prompt else ""
    adapter_suffix = "qlora_adapters" if args.use_qlora else "lora_adapters"
    adapter_output_dir = f"{save_path}/{args.num_epochs}/{adapter_suffix}{system_prompt_suffix}"
    merged_suffix = "qlora_merged" if args.use_qlora else "lora_merged"
    merged_output_dir = f"{save_path}/{args.num_epochs}/{merged_suffix}{system_prompt_suffix}"
    
    print("🔗 LoRA 어댑터 로드 및 병합...")
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_output_dir)
    model = model.merge_and_unload() 
    
    print("💾 병합된 모델 저장...")
    
    model.save_pretrained(merged_output_dir)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=args.cache_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(merged_output_dir)

    print(f"✅ 완료: {merged_output_dir}")
    
if __name__ == "__main__":
    main()