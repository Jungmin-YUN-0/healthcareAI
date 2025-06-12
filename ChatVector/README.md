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

### Extracting Chat Vector

* `extract_chat_vector.py` 파일을 참고
* Instruction Chat Vector: `beomi/Llama-3-KoEn-8B-Instruct-preview` - `beomi/Llama-3-KoEn-8B`
* Korean Chat Vector: `beomi/Llama-3-KoEn-8B` - `meta-llama/Meta-Llama-3-8B`
* OpenBioLLM Chat Vector: `aaditya/Llama3-OpenBioLLM-8B` - `meta-llama/Meta-Llama-3-8B`
* 각각의 Chat Vector를 얻기 위해서 미리 작성된 아래 스크립트를 실행
* 별개의 Chat Vector가 필요할 경우 스크립트 수정을 통해 Chat Vector를 얻을 수 있음

```bash
bash extract_IT.sh # Instruction Chat Vector
bash extract_KO.sh # Korean Chat Vector
bash extract_OB.sh # OpenBioLLM Chat Vector
```

### Adding Chat Vector

* `add_chat_vector.py` 파일을 참고
* 2개 이상의 Chat Vector를 더해주는 경우, `--ratio` 옵션을 통해 조절 가능 (`add_KO+IT+OB.sh` 참고)
* Korean Llama + OpenBioLLM: `beomi/Llama-3-KoEn-8B` + (`aaditya/Llama3-OpenBioLLM-8B` - `meta-llama/Meta-Llama-3-8B`)
* Korean Llama + 0.5 Instruction + 0.5 OpenBioLLM: `beomi/Llama-3-KoEn-8B` + 0.5(`beomi/Llama-3-KoEn-8B-Instruct-preview` - `beomi/Llama-3-KoEn-8B`) + 0.5(`aaditya/Llama3-OpenBioLLM-8B` - `meta-llama/Meta-Llama-3-8B`)
* OpenBioLLM + Korean Chat Vector: `aaditya/Llama3-OpenBioLLM-8B` + (`beomi/Llama-3-KoEn-8B` - `meta-llama/Meta-Llama-3-8B`)
* 미리 작성된 아래 스크립트를 실행
* 다른 조합 또는 비율이 필요할 경우 스크립트 수정을 통해 Chat Vector를 더한 모델을 얻을 수 있음

```bash
bash add_KO+OB.sh # Korean Llama + OpenBioLLM
bash add_KO+IT+OB.sh # Korean Llama + 0.5Instruction + 0.5OpenBioLLM
bash add_OB+KO.sh # OpenBioLLM + Korean Chat Vector
```

## Chat Script

* System Prompt: `chat.py` 파일을 참고하여 `SYS_PROMPT` 변수를 수정

```bash
python chat.py \
$MODEL_PATH \  # 만들어진 모델 경로 (예: ckpt/Llama-3-8B-OpenBioLLM-Korean)
```

## Other Files

* `chat_exaone.ipynb`: EXAONE 모델을 활용한 대화 데모 및 예시 파일
* `chat_llama.ipynb`: OpenBioLLM 모델을 활용한 대화 데모 및 예시 파일
