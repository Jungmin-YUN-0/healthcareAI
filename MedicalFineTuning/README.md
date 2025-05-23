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



# 05/24

## <RAPA 인수인계>
-	모든 폴더는 /nas_homes/projects/rapa에 위치.


-	폴더 별 설명
```bash
* 1)	Chatvector
-	Chatvector 모델 위치 (생성 방법 Github 참조)
* 2)	Dataset 
- A.	Blossom (blossom Fine-tuning 용)
- B.	Data_updated  = 의료 증강 데이터; PPT 참조 (서울대 + 기존 데이터)
- C.	Data_update_512 : 512 이를 512토큰 자름

* 3)	Results : 이때까지 한 모들 결과
- A.	BERT Score, BLEU 등등… -> CoT는 steering vector (https://github.com/Marker-Inc-Korea/COT_steering/tree/main)

* 4)	Code
- A.	Data_checker : 정민님이 주신 데이터 전처리 (위의 data_updated로 이미 완료되어짐)
- B.	LoRA_first.py : Openbiollm 모델 학습
Exaone.py : Exaone 모델 학습 코드 (생성한 모델은 용량 문제로 제거, 학습 필요)
- C.	Ds_confing.json : Deepspeed 코드
- D.	Generation1.ipynb : 실제 텍스트 생성 코드
- E.	나머지 폴더는 권한이 없어 제거하지 못함
```
