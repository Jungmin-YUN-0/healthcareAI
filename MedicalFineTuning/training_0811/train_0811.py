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
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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

def get_model_config(model_name):
    """
    모델별 설정을 반환하는 함수
    """
    model_configs = {
        "yanolja/EEVE-Korean-10.8B-v1.0": {
            "prompt_template": "Human: {prompt}\nAssistant: {response}",
            "stop_criteria": ["Human:", "\n\nHuman:", "Assistant:", "\n\nAssistant:"],
            "model_type": "eeve"
        },
        "trillionlabs/Tri-7B": {
            "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>",
            "stop_criteria": ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
            "model_type": "tri"
        },
        "skt/A.X-4.0-Light": {
            "prompt_template": "<s>[INST] {prompt} [/INST] {response}</s>",
            "stop_criteria": ["[INST]", "[/INST]", "<s>", "</s>"],
            "model_type": "ax"
        }
    }
    
    for model_key in model_configs:
        if model_key in model_name or model_name in model_key:
            return model_configs[model_key]
    
    # Default configuration for other models
    return {
        "prompt_template": "질문: {prompt}\n답변: {response}",
        "stop_criteria": ["질문:", "\n질문:", "답변:", "\n답변:"],
        "model_type": "default"
    }

def prompt_generate(example, system_prompt=None, use_system_prompt=False, output_key='output', model_name=None):
    """
    데이터셋을 프롬프트 형태로 변환하는 함수
    """
    input_text = example['input']
    output_text = example[output_key]
    
    # 모델별 설정 가져오기
    model_config = get_model_config(model_name) if model_name else get_model_config("default")
    
    if use_system_prompt and system_prompt:
        if model_config["model_type"] == "eeve":
            prompt = f"{system_prompt}\n\nHuman: {input_text}\nAssistant: {output_text}"
        elif model_config["model_type"] == "tri":
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
        elif model_config["model_type"] == "ax":
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
        else:
            prompt = f"{system_prompt}\n\n질문: {input_text}\n답변: {output_text}"
    else:
        if model_config["model_type"] == "eeve":
            prompt = f"Human: {input_text}\nAssistant: {output_text}"
        elif model_config["model_type"] == "tri":
            prompt = f"<|im_start|>system\nYou are Trillion, created by TrillionLabs. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
        elif model_config["model_type"] == "ax":
            prompt = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
        else:
            prompt = f"다음 질문에 대해서 답변을 생성하세요.\n\n질문: {input_text}\n답변: {output_text}"
    
    return {"text": prompt}


def tokenize_function(examples, tokenizer, max_length, model_name=None):
    """
    프롬프트 텍스트를 토크나이즈하고,
    프롬프트(질문) 부분을 -100으로 마스킹하여 라벨 생성
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
    model_config = get_model_config(model_name) if model_name else get_model_config("default")
    
    for i, text in enumerate(examples["text"]):
        # 모델별 프롬프트 부분 추출
        if model_config["model_type"] == "eeve":
            if "\nAssistant: " in text:
                prompt_part = text.split("\nAssistant: ")[0] + "\nAssistant: "
            else:
                prompt_part = text.split("\n")[0] + "\n"
        elif model_config["model_type"] == "tri":
            if "<|im_start|>assistant\n" in text:
                prompt_part = text.split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"
            else:
                # system과 user 부분까지 포함해서 마스킹
                if "<|im_start|>system" in text and "<|im_start|>user" in text:
                    prompt_part = text.split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"
                else:
                    prompt_part = text.split("<|im_end|>")[0] + "<|im_end|>\n"
        elif model_config["model_type"] == "ax":
            if "<|im_start|>assistant\n" in text:
                prompt_part = text.split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"
            else:
                if "<|im_start|>system" in text and "<|im_start|>user" in text:
                    prompt_part = text.split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"
                else:
                    prompt_part = text.split("<|im_end|>")[0] + "<|im_end|>\n"
        else:
            # Default behavior
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
        example_labels = tokenized["input_ids"][i].copy()
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
    parser.add_argument("--cache_path", type=str, default="/home/project/rapa/cache")
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=500)
    
    # Training arguments - Generation length
    parser.add_argument("--max_length", type=int, default=384)
    
    
    # Type arguments
    parser.add_argument("--type", type=int, choices=[1, 2, 3], required=True,
                       help="Training type: 1 (structured), 2 (shortened), 3 (shortened_50)")
    
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
    train_dataset = dataset['train'].shuffle(seed=args.random_seed)
    
    # Define system prompts based on type
    system_prompts = {
        1: '''## 응답 출력 지침
- 출력 형식: 아래 JSON 형식 그대로 사용 (추가 텍스트/설명/기호 금지)
{"_original": "정식 답변", "_short": "간단 요약 문장"}
- _original: **반드시 3문장 이하, 70자 이내**의 간결한 답변
- _short: **2문장 이하, 20자 이내, 핵심 내용 위주의 자연스러운 대화체 요약** (개조식 금지)
예시:
{"_original":"콜레스테롤 관리가 중요해요. 운동과 식단을 신경 써보세요. 추가로 궁금한 점 있으신가요?","_short":"콜레스테롤 관리가 중요해요."}

## 역할: 건강 전문 상담사 AI
- 친절하고 전문적이며, 누구나 이해할 수 있는 건강 정보 제공
- 진단이나 치료는 하지 않으며, 필요 시 의료 전문가나 진료과 안내

## 답변 지침
1. 길이 제한
- 모든 응답은 반드시 **3문장 이하, 70자 이내**로 제한
- 부연설명 없이, 짧고 명확하게 핵심 정보만 전달
2. 대화 유도 및 추가 질문
- 짧은 응답 후, 관련 후속 질문 또는 증상 구체화 질문 제시
- 예: "언제부터 이런 증상이 있었나요?"
3. 전문의 안내
- 필요한 경우에 한해, 의심 증상에 따라 해당 진료 과 권유
- 예: "이런 증상은 내과 진료가 도움이 될 수 있어요."
4. 범위 제한
- 건강과 무관한 질문은 정중히 불가 안내 후 건강 주제로 유도
- 예: "그 부분은 답변드리기 어려워요. 건강 관련 궁금한 점 있으신가요?"
5. 한국어 사용
- 항상 자연스럽고 정확한 한국어 사용
- 요청 없는 한 영어 및 전문 용어 사용 금지''',
        2: '''## 응답 출력 지침
- 출력 형식: 아래 JSON 형식 그대로 사용 (추가 텍스트/설명/기호 금지)
{"_answer": "정식 답변", "_follow_up": "추가 질문"}
- _answer: 3문장 이하, 반드시 **45자 이내** 문장은 자연스럽게 이어져야 하며, 불필요하게 잘라내거나 딱딱하게 표현하지 않음. (개조식 금지)
- _follow_up: 답변 흐름에 맞는 자연스럽고 부드러운 후속 질문 한 문장

## 역할: 건강 전문 상담사 AI
- 친절하고 전문적이며, 일상 대화처럼 친절하고 자연스러운 어투로 건강 정보를 제공
- 전문 용어는 꼭 필요한 경우에만 사용하고, 이해하기 쉬운 표현을 우선
- 진단이나 치료는 하지 않으며, 필요 시 관련 의료 전문가나 진료과 안내

## 답변 지침
1. 길이 제한
- _answer: 45자 이내, 3문장 이하
- _follow_up: 1문장
- 핵심 정보 전달을 우선하되, 문장 간 자연스러운 연결 유지
2. _answer 작성
- 핵심 내용을 짧고 명확하게, 부드러운 문장 흐름으로 작성
3. follow_up 작성
- 답변과 자연스럽게 이어지는 후속 질문 제시
- 예: "언제부터 그러셨나요?" / "조금 더 설명해주실래요?" / "좀 더 자세히 설명드릴까요?" 등
4. 전문의 안내
- 필요한 경우, 의심 증상에 따라 해당 진료 과를 간단히 안내
- 예: "내과 진료가 도움이 될 수 있어요."
5. 건강과 무관한 질문은 정중히 불가 안내 후 건강 주제로 유도
- 예: "그 부분은 답변드리기 어려워요. 건강 관련 궁금한 점 있으신가요?"
6. 언어 규칙
- 항상 자연스럽고 정확한 한국어 사용
- 요청 없는 한 영어·전문 용어 사용 금지''',
        3: '''## 역할: 건강 전문 상담사 AI
- 친절하고 전문적이면서도 일상 대화처럼 자연스럽게 건강 정보를 제공
- 전문 용어는 꼭 필요한 경우에만 사용하고, 이해하기 쉬운 표현을 우선
- 진단이나 치료는 하지 않으며, 필요 시 관련 의료 전문가나 진료과 안내

## 답변 지침
- 핵심 정보를 50자 내외로 짧고 명확하게, 부드러운 흐름으로 작성
- 필요 시 답변과 자연스럽게 이어지는 후속 질문 포함
- 건강과 무관한 질문은 정중히 불가 안내 후 건강 주제로 유도
- 요청 없는 한 항상 자연스럽고 정확한 한국어 사용'''
    }
    
    # Set system prompt based on type
    system_prompt = system_prompts[args.type]
    print(f"유형 {args.type}의 시스템 프롬프트를 사용합니다.")
    
    # Define output key based on type
    output_keys = {
        1: 'output_structured',
        2: 'output_shortened', 
        3: 'output_shortened_50'
    }
    output_key = output_keys[args.type]
    
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
    if args.base_model_name == "EEVE+ChatVector":
        model_name = 'yanolja/EEVE-Korean-10.8B-v1.0'
        model_path = "/home/project/rapa/ckpt/eeve_chatvector_IT_OB"
    elif args.base_model_name == 'eeve_inst_chatvector_OB':
        model_name = '/home/project/rapa/ckpt/eeve_inst_chatvector_OB'
        model_path = '/home/project/rapa/ckpt/eeve_inst_chatvector_OB'
    elif args.base_model_name == "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct":
        model_name = args.base_model_name
        model_path = args.base_model_name
    elif args.base_model_name == 'eeve_chatvector_OB_update':
        model_name = '/home/project/rapa/ckpt/eeve_chatvector_OB_update'
        model_path = '/home/project/rapa/ckpt/eeve_chatvector_OB_update'
    elif args.base_model_name == "yanolja/EEVE-Korean-10.8B-v1.0":
        model_name = 'yanolja/EEVE-Korean-10.8B-v1.0'
        model_path = "yanolja/EEVE-Korean-10.8B-v1.0"
    elif args.base_model_name == "trillionlabs/Tri-7B":
        model_name = 'trillionlabs/Tri-7B'
        model_path = "trillionlabs/Tri-7B"
    elif args.base_model_name == "skt/A.X-4.0-Light":
        model_name = 'skt/A.X-4.0-Light'
        model_path = "skt/A.X-4.0-Light"
    else:
        model_name = args.base_model_name
        model_path = args.base_model_name
    
    # 데이터셋을 프롬프트 형태로 변환
    columns_to_keep = ['input', output_key]
    columns_to_remove = [col for col in train_dataset.column_names if col not in columns_to_keep]
    
    train_dataset = train_dataset.map(
        lambda examples: prompt_generate(examples, system_prompt, True, output_key, model_name), 
        remove_columns=columns_to_remove,
        cache_file_name=None,
        load_from_cache_file=False
    )
    
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=args.cache_path,
        trust_remote_code=True
    )
    
    
    if (tokenizer.pad_token is None) and (tokenizer.eos_token is not None):
        tokenizer.pad_token = tokenizer.eos_token
    
    # 토크나이징 적용 (패딩까지 미리 적용)
    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length, model_name),
        batched=True,
        remove_columns=['text', 'input', output_key], 
        desc="토크나이징 및 라벨 마스킹"
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config if args.use_qlora else None,
        torch_dtype=torch.bfloat16 if args.torch_dtype_bf16 else torch.float16,
        trust_remote_code=True,    
        cache_dir=args.cache_path,
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
    
    # Generate run name based on model, LoRA type, and training type
    model_short_name = args.base_model_name
    lora_type = "QLoRA" if args.use_qlora else "LoRA"
    type_suffix = f"_v{args.type}"
    run_name = f"RAPA/{model_short_name}_{lora_type}{type_suffix}"
    
    if args.use_qlora:
        ds_config = f'{args.deepspeed_config_path}_stage2.json'
        checkpoint_path = f'{args.checkpoint_path}/{args.base_model_name}/qlora{type_suffix}'
    else:
        ds_config = f'{args.deepspeed_config_path}_stage3.json'
        checkpoint_path = f'{args.checkpoint_path}/{args.base_model_name}/lora{type_suffix}'
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
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
        data_collator=data_collator
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
    save_path = f'{args.save_path}/{args.base_model_name}'
    # Save LoRA adapters
    if save_path:
        # use_qlora 값에 따라 저장 경로 설정
        adapter_suffix = "qlora_adapters" if args.use_qlora else "lora_adapters"
        adapter_output_dir = f"{save_path}/{args.num_epochs}/{adapter_suffix}{type_suffix}"
        # 1. 어댑터만 저장 (작은 용량)
        trainer.save_model(adapter_output_dir)
        tokenizer.save_pretrained(adapter_output_dir)
        print(f"어댑터와 토크나이저가 {adapter_output_dir}에 저장되었습니다.")

 
if __name__ == "__main__":
    main()
