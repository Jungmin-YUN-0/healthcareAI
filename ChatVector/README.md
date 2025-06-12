# Chat Vector를 통한 OpenBioLLM-Korean (Preview) 생성

[Chat Vector: A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages](https://arxiv.org/pdf/2310.04799).

## Requirements

* torch
* transformers
* fire

아래와 같은 스크립트를 통해 필요한 라이브러리를 설치

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118 # change to your CUDA version
pip install transformers fire
```

## Usage

### `extract_chat_vector.py`

*  파일을 참고
* Korean Chat Vector: KoLLM - base LLM (e.g. `beomi/Llama-3-KoEn-8B` - `meta-llama/Meta-Llama-3-8B`)
* OpenBioLLM Chat Vector: Medical LLM - base LLM (e.g. `aaditya/Llama3-OpenBioLLM-8B` - `meta-llama/Meta-Llama-3-8B`)

```bash
bash extract_KO.sh # Korean Chat Vector
bash extract_OB.sh # Medical Chat Vector
```

### `add_chat_vector.py`

* 2개 이상의 Chat Vector를 더해주는 경우, `--ratio` 옵션을 통해 조절 가능
* KoLLM + Medical Chat Vector (e.g. `beomi/Llama-3-KoEn-8B` + (`aaditya/Llama3-OpenBioLLM-8B` - `meta-llama/Meta-Llama-3-8B`))
* Medical LLM + Korean Chat Vector (e.g. `aaditya/Llama3-OpenBioLLM-8B` + (`beomi/Llama-3-KoEn-8B` - `meta-llama/Meta-Llama-3-8B`))
* 다른 조합 및 비율 설정 가능

```bash
bash add_KO+OB.sh
bash add_OB+KO.sh
```

## `chat.py`

* System Prompt: `chat.py` 파일을 참고하여 `SYS_PROMPT` 변수를 수정

```bash
python chat.py \
$MODEL_PATH \  # 만들어진 모델 경로 (예: ckpt/Llama-3-8B-OpenBioLLM-Korean)
```

## Other Files

* `chat_llama.ipynb`: OpenBioLLM 모델을 활용한 대화 데모 및 예시 파일
