# %% [markdown]
# # Dialogue generation Evaluation
# 
# - Dialogue 생성에 대한 Evaluation
# - Client - Doctor - Client 이후에 Doctor의 
# - BERTScore / Rouge score 등을 계산
# 
# 
# # QA EValuation
# - QA 생성에 대한 Evaluation
# - A / B / C / D / E 중 선택하도록 설 & Accuracy / Recall / Precision / F1 Score 등을 계산
# 

# %%
import torch
import torch.nn as nn
import torch.distributed as dist


import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score


# %%


# %%
from vllm import LLM, SamplingParams
import pandas as pd
import torch
import torch._dynamo



torch._dynamo.config.suppress_errors = True

# %%
cleaned_dataset = load_from_disk('/nas_homes/projects/rapa/Dataset/cleaned_dataset_with_answer_tag_new')

# %% [markdown]
# ## Dialogue Generation Evaluation

# %%
dialogue = [i['input'] for i in cleaned_dataset['test'] if i['source'] == 'dialogue']

# %%
print(dialogue[0])

# %%
y_true = [i['output'] for i in cleaned_dataset['test'] if i['source'] == 'dialogue']

# %% [markdown]
# ### vLLM 사용

# %%
# vLLM 모델 로드 (GPU 지원)
model_name = "YOUR_MODEL_NAME"
# LLM 초기화 (디버깅을 위해 단일 GPU, 비동기 비활성화)
llm = LLM(
    model= model_name
    # tensor_parallel_size=1,  # 안정성 위해 GPU 1개만 사용
)

# Sampling parameters 설정
sampling_params = SamplingParams(temperature=0.0, max_tokens = 512)

print("before")
# vLLM을 사용한 병렬 추론
outputs = llm.generate(dialogue, sampling_params)

print("after")

# 정답 및 예측값 저장
y_pred = outputs


# %%
y_pred_real = [i.outputs[0].text for i in y_pred]

# %%
y_true_real =  y_true

# %%
# 결과 출력
for i, (true_ans, pred_ans) in enumerate(zip(y_true, y_pred_real)):
    print(f"Q{i+1}: True: {true_ans}, / {pred_ans}")

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %% [markdown]
# ### Dialogue Generation Performance
# 

# %%

# 성능 평가 계산

def exact_match_score(y_true_real, y_pred_real):
    return sum([1 if pred.lower() == true.lower() else 0 for pred, true in zip(y_pred_real, y_true_real)]) / len(y_true_real)

exact_match = exact_match_score(y_true_real, y_pred_real)

# Hugging Face 토크나이저로 토큰화
def tokenize_text(text):
    return tokenizer.tokenize(text)

bleu_scores = [sentence_bleu([tokenize_text(true)], tokenize_text(pred)) for true, pred in zip(y_true_real, y_pred_real)]
avg_bleu = sum(bleu_scores) / len(bleu_scores)

rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
rouge_scores = [rouge.score(true, pred) for true, pred in zip(y_true_real, y_pred_real)]
avg_rouge1 = sum([score["rouge1"].fmeasure for score in rouge_scores]) / len(rouge_scores)
avg_rouge2 = sum([score["rouge2"].fmeasure for score in rouge_scores]) / len(rouge_scores)
avg_rougeL = sum([score["rougeL"].fmeasure  for score in rouge_scores]) / len(rouge_scores)

# METEOR Score 수정 (Hugging Face 토크나이저 사용)
meteor_scores = [meteor_score([tokenize_text(true)], tokenize_text(pred)) for true, pred in zip(y_true_real, y_pred_real)]
avg_meteor = sum(meteor_scores) / len(meteor_scores)

# 결과 출력
print("Exact Match Score:", exact_match)
print("BLEU Score:", avg_bleu)
print("ROUGE-1 Score:", avg_rouge1)
print("ROUGE-2 Score:", avg_rouge2)
print("ROUGE-L Score:", avg_rougeL)
print("METEOR Score:", avg_meteor)    

# %%
df_all = pd.DataFrame([avg_bleu, avg_rouge1, avg_rouge2, avg_rougeL, avg_meteor]).T


df_all.columns = ['bleu', 'Rouge1', 'Rouge2', 'RougeL', 'Meteor']

# %%
from bert_score import score
import torch

P, R, F1 = score(
    y_pred_real, y_true_real,
    model_type='klue/bert-base',
    num_layers=12,
    lang='ko',
    verbose=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)


df_score = pd.DataFrame([F1])
df_score.columns = ['BERTscore']

df_total = pd.concat([df_score, df_all], axis = 0)

df_total.to_csv(f'/nas_homes/projects/rapa/Result/{model_name}_Dialogue_Generation_Results.csv', encoding = 'utf-8-sig')

