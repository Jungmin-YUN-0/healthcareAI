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

  `KoAlpaca_v1.1_medical.jsonl`: KoAlpaca 데이터셋에서 의료 관련 질문 및 답변만을 추출한 데이터셋

## Finetuning

* `--data_path`: 학습에 사용할 데이터셋 경로 (JSONL 형식).
* `--base_model`: 사전 학습된 한국어 언어 모델 경로.
* `--output_dir`: 학습 과정에서의 Output을 저장할 경로
* 학습된 모델은 `--base_model`과 같은 경로에 `-InstructFT`가 추가된 이름으로 저장됨

```bash
python script.py --data_path './KoAlpaca_v1.1_medical.jsonl' --base_model '../ChatVector/ckpt/Llama-3-8B-OpenBioLLM-Korean' --output_dir './outputs'
```



# 05/24

## <RAPA 인수인계>
-	모든 폴더는 /nas_homes/projects/rapa에 위치.

-	
# 📁 프로젝트 폴더 구조 안내

본 저장소의 주요 폴더 및 파일 구조에 대한 설명입니다. 각 폴더는 모델 학습, 데이터셋 관리, 결과 확인 등을 위해 구성되어 있습니다.

---

## 1. `Chatvector/`
Chatvector 관련 모델 파일이 위치한 디렉토리입니다.  
- Chatvector 모델 저장  
- 생성 방법은 [공식 GitHub 저장소](https://github.com/Marker-Inc-Korea/COT_steering/tree/main) 참고

---

## 2. `Dataset/`
모델 학습 및 파인튜닝에 사용되는 데이터셋들이 포함된 디렉토리입니다.

- **`Blossom/`**  
  Blossom 모델 파인튜닝용 데이터셋

- **`Data_updated/`**  
  의료 증강 데이터셋 (서울대학교 + 기존 데이터 혼합)  
  ※ 상세한 내용은 내부 PPT 참고

- **`Data_update_512/`**  
  `Data_updated` 데이터를 512 토큰 단위로 분할한 버전

---

## 3. `Results/`
지금까지 진행된 모델 실험 결과를 저장하는 디렉토리입니다.

- BERT Score, BLEU 등 다양한 평가 지표 포함  
- CoT (Chain-of-Thought) 결과는 [Steering Vector 저장소](https://github.com/Marker-Inc-Korea/COT_steering/tree/main) 참고

---

## 4. `Code/`
데이터 전처리, 모델 학습 및 텍스트 생성 등에 관련된 코드가 포함되어 있습니다.

- **`Data_checker/`**  
  정민님이 제공한 데이터 전처리 코드  
  (`Data_updated`에 이미 반영 완료)

- **`LoRA_first.py`**  
  OpenBioLLM 모델 학습 코드

- **`Exaone.py`**  
  Exaone 모델 학습 코드  
  ※ 모델 파일은 용량 문제로 삭제됨 → 재학습 필요

- **`ds_config.json`**  
  Deepspeed 설정 파일

- **`Generation1.ipynb`**  
  실제 텍스트 생성 코드

- **기타 폴더**  
  권한 문제로 인해 제거하지 못한 폴더 일부 존재
