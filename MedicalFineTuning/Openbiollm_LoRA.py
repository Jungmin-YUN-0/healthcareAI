# %%
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel
import wandb

# 환경 세팅

# %%
# 모델 및 데이터셋
model_name = "openbiollm_chatvector_new"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 필수 설정

dataset = load_from_disk("/nas_homes/projects/rapa/data_updated_512_instruction")

# %%
# 모델 로딩
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# 기존 체크포인트가 있다면 여기서 이어붙이기 가능
# model = PeftModel.from_pretrained(model, "your_checkpoint_path")

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# %%
# wandb init
wandb.init(
    project="lora-finetune-chatvector",
    name="lora-basic-tuning"
)

# %%
# 훈련 설정
training_args = TrainingArguments(
    output_dir="checkpoints_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    bf16=True,
    evaluation_strategy="no",   # ✅ 평가 안 함
    save_strategy="epoch",      # 필요시 "steps"로 바꿔도 됨
    logging_steps=10,
    save_total_limit=1,
    report_to="wandb",
    run_name="lora-basic-tuning",
    remove_unused_columns=False,
    gradient_checkpointing = True
)


# %%
# 데이터 collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# %%
# Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# %%
# 학습 시작
trainer.train()

# %%
# 병합 및 저장
model.merge_and_unload()
output_dir = "checkpoints_lora_merged"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Merged full model saved at {output_dir}")
