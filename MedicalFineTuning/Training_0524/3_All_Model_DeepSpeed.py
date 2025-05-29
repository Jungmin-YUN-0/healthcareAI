# %%
from datasets import load_from_disk
import torch
import matplotlib.pyplot as plt
from datasets import DatasetDict
from transformers import AutoTokenizer
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import AdaLoraConfig, get_peft_model, TaskType
from datasets import DatasetDict


model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

# 예시 tokenizer (필요에 따라 교체)
tokenizer = AutoTokenizer.from_pretrained(model_name)  # 또는 원하는 모델로 변경



# -----------------------------
# 🟡 0. WandB 설정
# -----------------------------
wandb_project_name = "exaone-adalora-finetune"  # ← 여기만 네가 원하는 이름으로 바꾸면 돼
wandb_run_name = "EXAONE-output-only-run"       # 선택 사항
wandb.init(project=wandb_project_name, name=wandb_run_name)

# -----------------------------
# 🟡 1. Load model/tokenizer
# -----------------------------
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
)


# %%
# -----------------------------
# 🟡 2. PEFT - AdaLoRA
# -----------------------------

# %%
tokenized_datasets = load_from_disk('/nas_homes/projects/rapa/Dataset/filtered_updated_datasets_512_instruction')

train_size = len(tokenized_datasets["train"])
batch_size = 4
grad_accum = 4
epochs = 5
total_step = (train_size // (batch_size * grad_accum)) * epochs

peft_config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    init_r=8,
    target_modules=["q_proj", "v_proj"],  # 가장 영향 큰 projection 모듈만
    lora_alpha=32,
    lora_dropout=0.1,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    beta1=0.85,
    beta2=0.85,
    total_step=total_step,  # 👈 반드시 추가!
)


model = get_peft_model(model, peft_config)



# %%
from transformers import default_data_collator

# -----------------------------
# 🟡 5. TrainingArguments (WandB 설정 포함)
# -----------------------------
training_args = TrainingArguments(
    output_dir="exaone_adalora_ft",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    num_train_epochs=epochs,

    evaluation_strategy="epoch",      # ✅ epoch마다 평가
    save_strategy="epoch",            # ✅ epoch마다 저장
    save_total_limit = 1,
    logging_steps=1,                  # 계속 남기고 싶으면 유지
    learning_rate=5e-5,
    bf16=True,
    fp16=False,
    torch_compile=False,
    logging_dir="logs",
    report_to="wandb",
    run_name=wandb_run_name,
    deepspeed="ds_config.json",
)


# -----------------------------
# 🟡 6. Trainer + 학습 실행
# -----------------------------
data_collator = default_data_collator

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

model.merge_and_unload()
output_dir = "/nas_homes/projects/rapa/Models/exaone_adalora_ft" # nas 모델 저장 경로
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Merged full model saved at {output_dir}")