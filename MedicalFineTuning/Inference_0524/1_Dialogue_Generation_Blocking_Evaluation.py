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
from vllm import LLM, SamplingParams
import pandas as pd
import torch
import torch._dynamo
from datasets import load_from_disk, DatasetDict, Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
import re
from bert_score import score
import torch

# %%

dataset = load_from_disk("cleaned_dataset_with_answer_tag_new") # Preprocessing에서 저장한 데이터 셋

# %%
dataset

# %%
print(dataset['test']['input'][0])

# %%
question = []
y_true = []
for input, source, output in zip(dataset['test']['input'], dataset['test']['source'], dataset['test']['output']):
    if source == 'dialogue':
        question.append(input)
        y_true.append(output)
    

# %%
print(question[1])

# %%


## Blockin English, Chinese, Japanese + 특수문자 Unicdoe
class ForeignLangWithEnglishOnceBlocker(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.foreign_lang_mask = None
        self.english_seen = False  # 영어가 한 번이라도 등장했는지 추적

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = scores.device

        # 토큰 전체 디코딩 → 영어가 나왔는지 확인
        decoded_so_far = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        full_text = " ".join(decoded_so_far)

        if not self.english_seen and re.search(r'[a-zA-Z]', full_text):
            self.english_seen = True  # 영어가 처음 등장함

        # foreign_lang_mask 처음 초기화
        if self.foreign_lang_mask is None:
            token_ids = torch.arange(scores.size(-1), device=device)
            decoded_tokens = self.tokenizer.batch_decode(token_ids.unsqueeze(1), skip_special_tokens=True)

            # Unicode 범위 정의
            chinese_ranges = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0xF900, 0xFAFF)]
            japanese_ranges = [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)]
            russian_ranges = [(0x0400, 0x04FF), (0x0500, 0x052F)]
            special_char_ranges = [(0x2000, 0x206F),  # 일반 특수문자 (예: ‘ ” …)
                                   (0x2190, 0x21FF),  # 화살표
                                   (0x2300, 0x23FF),  # 기술 기호
                                   (0x2500, 0x257F),  # 박스 드로잉
                                   (0x25A0, 0x25FF),  # 기호
                                   (0x2B00, 0x2BFF),  # 기타 기호
                                   (0x1F000, 0x1F9FF)]  # 이모지 등

            all_ranges = chinese_ranges + japanese_ranges + special_char_ranges

            def is_foreign(token):
                return any(
                    any(start <= ord(c) <= end for start, end in all_ranges)
                    for c in token if c.strip()
                )

            def is_english(token):
                return any(c.isascii() and c.isalpha() for c in token)

            # 두 개의 마스크 생성
            self.foreign_lang_mask = torch.tensor([is_foreign(t) for t in decoded_tokens], device=device)
            self.english_mask = torch.tensor([is_english(t) for t in decoded_tokens], device=device)

        # 무조건 차단할 외국어 (중국어, 일본어, 러시아어)
        scores[:, self.foreign_lang_mask] = -float("inf")

        # 영어는 아직 등장 안 했으면 차단
        if not self.english_seen:
            scores[:, self.english_mask] = -float("inf")

        return scores




# %%

# 1. 토크나이저 및 모델 로드
model_name = "YOUR_MODEL_NAME" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# %%
# 배치 추론 함수
batch_size=2
max_new_tokens=128
all_outputs = []

# %%

def make_custom_logits_filter(tokenizer):
    """
    ForeignLangWithEnglishOnceBlocker를 logits processor 리스트로 감싸서 반환하는 함수.
    """
    processors = LogitsProcessorList()
    processors.append(ForeignLangWithEnglishOnceBlocker(tokenizer))
    return processors


# %%
# 배치 추론 함수
batch_size=2
max_new_tokens=128
all_outputs = []
for i in tqdm(range(0, len(question), batch_size)):
    
    batch = question[i:i+batch_size]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")

    logits_processor = LogitsProcessorList()
    logits_processor.append(make_custom_logits_filter(tokenizer))

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=logits_processor,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    all_outputs.extend(decoded)


# %%
# pd.DataFrame(all_outputs).to_csv('임시.csv', encoding = 'utf-8-sig')

# %%
len(all_outputs), len(question)

# %% [markdown]
# # 질문 누락 확인
# 
# - Batch 단위로 수행하기 때문에 안 맞는 경우에 대응

# %%
mismatched_question_indices = []

for i in range(len(all_outputs)):
    out = all_outputs[i]
    q_curr = question[i]
    
    if len(out) >= len(q_curr) and out[:len(q_curr)] == q_curr:
        continue  # 정상 매칭

    if i + 1 < len(question):
        q_next = question[i + 1]
        if len(out) >= len(q_next) and out[:len(q_next)] == q_next:
            continue  # 다음 질문과 매칭됨

    # 매칭 실패 → question[i]가 매칭 안 됐다고 간주
    mismatched_question_indices.append(i)

print(f"총 mismatched question 개수: {len(mismatched_question_indices)}")
print(f"Mismatch된 질문 인덱스 예시: {mismatched_question_indices[:10]}")


# %%


# %%
index_to_redo = 1975
q = question[index_to_redo]

inputs = tokenizer([q], return_tensors="pt", padding=True, truncation=True).to("cuda")

logits_processor = LogitsProcessorList()
logits_processor.append(make_custom_logits_filter(tokenizer))

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        logits_processor=logits_processor,
    )

decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
new_output = decoded[0]

# %%
all_outputs.insert(1974, new_output)


# %%
# 마지막 질문이 마지막 출력에 포함됐는지 확인
last_q = question[-1]
last_out = all_outputs[-1]

if len(last_out) < len(last_q) or last_out[:len(last_q)] != last_q:
    print("⚠️ 마지막 질문도 출력에 포함되지 않았음!")
    print("추가로 하나 더 누락된 질문이 존재합니다.")
    print(f"질문 인덱스: {len(question) - 1}")
    print(f"질문 내용: {last_q[:100]}...")
else:
    print("✅ 마지막 질문은 정상적으로 포함되어 있음.")


# %%
# 누락된 질문 인덱스
missing_index = 1975

# 새로운 output을 해당 위치에 삽입하되,
# 그 이전까지는 그대로, 이후는 원래보다 하나씩 뒤로 shift해서 매칭되도록 함
corrected_outputs = (
    all_outputs[:missing_index] +       # 앞부분은 문제 없음
    [new_output] +                      # 누락된 답변 추가
    all_outputs[missing_index:]         # 뒤는 원래보다 하나씩 앞으로 당겨서 매칭됨
)

# %%
len(corrected_outputs), len(question)

# %%
len(corrected_outputs), len(question)

# %%


bad_index = 1976
q = question[bad_index]

# 토크나이징
inputs = tokenizer([q], return_tensors="pt", padding=True, truncation=True).to("cuda")

# logits processor
logits_processor = LogitsProcessorList()
logits_processor.append(make_custom_logits_filter(tokenizer))

# 모델 추론
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        logits_processor=logits_processor,
    )

# 디코딩
decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
corrected_outputs[bad_index] = decoded[0]  # 대체!

print(f"✅ index {bad_index}의 출력 재생성 완료 및 덮어쓰기 완료!")


# %%
mismatched_question_indices = []

for i in range(len(corrected_outputs)):
    out = corrected_outputs[i]
    q_curr = question[i]
    
    if len(out) >= len(q_curr) and out[:len(q_curr)] == q_curr:
        continue

    if i + 1 < len(question):
        q_next = question[i + 1]
        if len(out) >= len(q_next) and out[:len(q_next)] == q_next:
            continue

    mismatched_question_indices.append(i)

print(f"총 mismatched question 개수: {len(mismatched_question_indices)}")
print(f"Mismatch된 질문 인덱스 예시: {mismatched_question_indices[:10]}")


# %%
y_pred_real = all_outputs

# %%
y_true_real =  y_true

# %%
y_pred_real[0]

# %%
y_pred_real_real = [i for i in y_pred_real]

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
y_pred_real = corrected_outputs

# %%
ffff = [i[len(j):].strip() for i,j in zip(y_pred_real,question)]

# %% [markdown]
# # Performance

# %%

# 성능 평가 계산

def exact_match_score(y_true_real, ffff):
    return sum([1 if pred.lower() == true.lower() else 0 for pred, true in zip(ffff, y_true_real)]) / len(y_true_real)

exact_match = exact_match_score(y_true_real, ffff)

# Hugging Face 토크나이저로 토큰화
def tokenize_text(text):
    return tokenizer.tokenize(text)

bleu_scores = [sentence_bleu([tokenize_text(true)], tokenize_text(pred)) for true, pred in zip(y_true_real, ffff)]
avg_bleu = sum(bleu_scores) / len(bleu_scores)

rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
rouge_scores = [rouge.score(true, pred) for true, pred in zip(y_true_real, ffff)]
avg_rouge1 = sum([score["rouge1"].fmeasure for score in rouge_scores]) / len(rouge_scores)
avg_rouge2 = sum([score["rouge2"].fmeasure for score in rouge_scores]) / len(rouge_scores)
avg_rougeL = sum([score["rougeL"].fmeasure  for score in rouge_scores]) / len(rouge_scores)

# METEOR Score 수정 (Hugging Face 토크나이저 사용)
meteor_scores = [meteor_score([tokenize_text(true)], tokenize_text(pred)) for true, pred in zip(y_true_real, ffff)]
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

# BERTScore 계산
P, R, F1 = score(
    ffff, y_true_real,
    model_type='klue/bert-base',
    num_layers=12,
    lang='ko',
    verbose=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)


# %%
df_score = pd.DataFrame([F1])
df_score.columns = ['BERTscore']

# %%
df_total = pd.concat([df_score, df_all], axis = 0)

df_total.to_csv(f'/nas_homes/projects/rapa/Result/{model_name}_Dialogue_Generation_Results.csv', encoding = 'utf-8-sig')


