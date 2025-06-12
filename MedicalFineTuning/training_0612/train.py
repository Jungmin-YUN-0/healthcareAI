import argparse
import torch
import os
import re
from datasets import load_from_disk
from transformers import (
    BitsAndBytesConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

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

def prompt_generate(example):
    """
    데이터셋을 프롬프트 형태로 변환하는 함수
    """
    input_text = example['input']
    output_text = example['output']
    
    prompt = f"다음 질문에 대해서 답변을 생성하세요.\n\n질문: {input_text}\n답변: {output_text}"
    
    return {"text": prompt}

def tokenize_function(examples, tokenizer, max_length):
    """
    텍스트를 토크나이징하고 프롬프트 부분을 마스킹하는 함수
    미리 패딩까지 처리하여 완전한 형태로 반환
    """
    # 전체 텍스트 토크나이징 (패딩 포함)
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,
        add_special_tokens=True
    )
    
    # labels 리스트 초기화
    labels = []
    
    # 각 예제에 대해 프롬프트 부분 마스킹
    for i, text in enumerate(examples["text"]):
        # 프롬프트 부분 추출 (질문 부분)
        if "\n답변: " in text:
            prompt_part = text.split("\n답변: ")[0] + "\n답변: "
        else:
            prompt_part = text.split("\n")[0] + "\n"
        
        # 프롬프트 부분 토크나이징 (길이 계산용)
        prompt_tokens = tokenizer(
            prompt_part,
            add_special_tokens=True,
            return_tensors=None
        )
        prompt_length = len(prompt_tokens["input_ids"])
        
        # 전체 input_ids에서 labels 생성 (복사)
        example_labels = tokenized["input_ids"][i][:]
        
        # 프롬프트 부분을 -100으로 마스킹
        for j in range(min(prompt_length, len(example_labels))):
            example_labels[j] = -100
            
        # 패딩 토큰도 -100으로 마스킹
        for j in range(len(example_labels)):
            if tokenized["input_ids"][i][j] == tokenizer.pad_token_id:
                example_labels[j] = -100
                
        labels.append(example_labels)
    
    # tokenized 결과에 labels 추가
    tokenized["labels"] = labels
    
    return tokenized

def main():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--base_model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--save_path", type=str, default="/home/project/rapa/final_model")
    parser.add_argument("--checkpoint_path", type=str, default="/home/project/rapa/checkpoint")
    parser.add_argument("--data_path", type=str, default="/home/project/rapa/dataset/dataset_fintuning_0603")
    
    # Deepspeed arguments
    parser.add_argument("--deepspeed_config_path", type=str, default='./ds_config')
    parser.add_argument("--use_deepspeed", type=bool, default=False)
    parser.add_argument("--local_rank", type=int, default=-1)
    
    # Training arguments - LoRA & QLoRA
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_qlora", type=str2bool, default=True)
    
    # Training arguments - Hyperparameters
    parser.add_argument('--job', type=str, default='training', choices=['training', 'resume_training'])
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=500)
    
    # Training arguments - Generation length
    parser.add_argument("--max_length", type=int, default=512)
    
    # Other arguments
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--torch_dtype_bf16", type=str2bool, default=True)
    parser.add_argument("--use_gradient_checkpointing", type=str2bool, default=True)
    
    args = parser.parse_args()
    
    os.environ["WANDB_PROJECT"] = 'RAPA'
    os.environ["WANDB_DISABLED"] = "false"
    
    # Set random seed
    torch.manual_seed(args.random_seed)
    if args.use_deepspeed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    # Load dataset
    dataset = load_from_disk(args.data_path)
    train_dataset = dataset['train']
    
    # 데이터셋을 프롬프트 형태로 변환
    train_dataset = train_dataset.map(
        prompt_generate, 
        remove_columns=['input', 'output', 'source'],
        cache_file_name=None,
        load_from_cache_file=False
    )
    
    # Configure quantization if using QLoRA
    quantization_config = None
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.torch_dtype_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # Load model and tokenizer based on model name
    model_path = args.base_model_name
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if (tokenizer.pad_token is None) and (tokenizer.eos_token is not None):
        tokenizer.pad_token = tokenizer.eos_token
    
    # 토크나이징 적용
    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=['text'],
        cache_file_name=None,
        load_from_cache_file=False,
        desc="토크나이징 및 라벨 마스킹"
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config if args.use_qlora else None,
        torch_dtype=torch.bfloat16 if args.torch_dtype_bf16 else torch.float16,
        trust_remote_code=True,    
    )

    # Prepare model for training
    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.use_gradient_checkpointing)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 일반적인 DataCollator 사용 (MLM=False로 설정하여 causal LM용으로)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    # Generate run name based on model and LoRA type
    model_short_name = args.base_model_name
    if '/home' in model_short_name:
        model_short_name = model_short_name.split('/')[-1] 
    lora_type = "QLoRA" if args.use_qlora else "LoRA"
    run_name = f"RAPA/{model_short_name}_{lora_type}"
    
    if args.use_qlora:
        ds_config = f'{args.deepspeed_config_path}_stage2.json'
        checkpoint_path = f'{args.checkpoint_path}/{model_short_name}/qlora'

    else:
        ds_config = f'{args.deepspeed_config_path}_stage3.json'
        checkpoint_path = f'{args.checkpoint_path}/{model_short_name}/lora'
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_strategy="epoch",                     # 매 epoch 끝마다 저장
        logging_steps=500,
        bf16=args.torch_dtype_bf16,
        remove_unused_columns=False,
        report_to='wandb',
        run_name=run_name,
        do_train=True,
        deepspeed=ds_config if args.use_deepspeed else None,
    )
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,  # 일반적인 collator 사용
    )
    
    # Start training
    if args.job == 'resume_training':
        checkpoint_root = checkpoint_path

        if os.path.exists(checkpoint_root):
            # checkpoint-* 디렉토리들 중에서 숫자 가장 큰 거 찾기
            checkpoint_dirs = [d for d in os.listdir(checkpoint_root) if re.match(r'checkpoint-\d+', d)]
            if checkpoint_dirs:
                latest_checkpoint = max(
                    checkpoint_dirs,
                    key=lambda x: int(x.split('-')[-1])
                )
                resume_path = os.path.join(checkpoint_root, latest_checkpoint)
                print(f"🔁 가장 최근 체크포인트에서 재개: {resume_path}")
                trainer.train(resume_from_checkpoint=resume_path)
            else:
                print("❌ checkpoint 디렉토리에는 checkpoint-* 폴더가 없습니다. 새로 학습을 시작합니다.")
                trainer.train()
        else:
            print("❌ checkpoint 경로가 존재하지 않습니다. 새로 학습을 시작합니다.")
            trainer.train()
    else:
        trainer.train()
    print('Train Finish')
    
    save_path = f'{args.save_path}/{model_short_name}'
    
    # Save LoRA adapters
    if save_path:
        # use_qlora 값에 따라 저장 경로 설정
        adapter_suffix = "qlora_adapters" if args.use_qlora else "lora_adapters"
        adapter_output_dir = f"{save_path}/{args.num_epochs}/{adapter_suffix}"
        # 1. 어댑터만 저장 (작은 용량)
        trainer.save_model(adapter_output_dir)
        tokenizer.save_pretrained(adapter_output_dir)
        print(f"어댑터와 토크나이저가 {adapter_output_dir}에 저장되었습니다.")

 
if __name__ == "__main__":
    main()
