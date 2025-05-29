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

# API KEY를 환경변수로 관리하기 위한 설정 파일
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"]= "1"


# %%
from vllm import LLM, SamplingParams
import pandas as pd
import torch
import torch._dynamo



torch._dynamo.config.suppress_errors = True

# %%
cleaned_dataset = load_from_disk('/nas_homes/projects/rapa/Dataset/cleaned_dataset_with_answer_tag_new')

# %% [markdown]
# ## QA Evaluation

# %%
question = [i['input'] for i in cleaned_dataset['test'] if i['source'] == 'qa']

# %%
y_true = [int(i['output'].replace(',','')) for i in cleaned_dataset['test'] if i['source'] == 'qa']

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
outputs = llm.generate(question, sampling_params)

print("after")

# 정답 및 예측값 저장
y_pred = outputs


# %%
y_pred_lst = []

# %%
for answer_text in y_pred:
    number = re.search(r'\d+', answer_text)
    if number:
        answer_number = int(number.group())
        y_pred_lst.append(answer_number)  # 출력: 5

# %%


# Accuracy
acc = accuracy_score(y_true, y_pred_lst)

# Precision, Recall, F1 (다중 클래스라면 average 방식 지정 필요)
precision = precision_score(y_true, y_pred, average='macro')  # 또는 'weighted', 'micro'
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# %%
df_qa_final = pd.DataFrame([acc,precision, recall, f1]).T

# %%
df_qa_final.columns = ['accuracy', 'precision', 'recall', 'f1score']

# %%
df_qa_final.to_csv(f'/nas_homes/projects/rapa/Result/{model_name}_QA_Generation_Results.csv', encoding = 'utf-8-sig')

# %%



