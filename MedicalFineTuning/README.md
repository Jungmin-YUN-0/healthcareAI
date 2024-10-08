# OpenBioLLM-Korean (Preview)를 대상으로 한 한국어 Finetuning

## Requirements

* torch
* transformers
* fire

아래와 같은 스크립트를 통해 필요한 라이브러리를 설치

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118 # change to your CUDA version
pip install pandas datasets transformers peft trl huggingface_hub bitsandbytes
```

## Data

* `KoAlpaca_v1.1_medical.jsonl`: KoAlpaca 데이터셋에서 의료 관련 질문 및 답변만을 추출한 데이터셋

## Finetuning

* `--data_path`: 학습에 사용할 데이터셋 경로 (JSONL 형식).
* `--base_model`: 사전 학습된 한국어 언어 모델 경로.
* `--output_dir`: 학습 과정에서의 Output을 저장할 경로
* 학습된 모델은 `--base_model`과 같은 경로에 `-InstructFT`가 추가된 이름으로 저장됨

```bash
python script.py --data_path './KoAlpaca_v1.1_medical.jsonl' --base_model '../ChatVector/ckpt/Llama-3-8B-OpenBioLLM-Korean' --output_dir './outputs'
```
