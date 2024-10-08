# RAG(Retrieval Augmented Generation) + ChatGPT

## Requirements
* torch
* transformers
* tqdm
* jsonlines
* rank-bm25
* openai

아래와 같은 스크립트를 통해 필요한 라이브러리를 설치

```bash
pip install torch transformers tqdm jsonlines rank-bm25
pip install openai==0.28.0
```

## Usage (w/Demo dataset)
### 1. Download dataset
* AI허브 초거대 AI 헬스케어 질의응답 데이터([Link](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71762))
* 데이터 다운로드 후, 작업 폴더 내에서 압출 풀기

* 압축 해제시, `1.질문`, `2.답변` 폴더 존재
* 본 Demo에서는 `1.질문`의 일부를 query, `2.답변` 전체를 검색 대상 corpus로 사용


### 2. Preprocess query and corpus
query 및 corpus 데이터셋을 얻기 위해 압축 해제한 raw dataset을 preprocess
`1.질문`, `2.답변` 폴더의 하위 질병 카테고리 (ex.감염성질환, 귀코목질환, 근골격질환 등)에서 원하는 질환 카테고리 선택
아래와 같이 preprocess_dataset.py로 raw dataset preprocess 후 jsonl 파일로 저장 (ex.소화기질환)

```bash
python preprocess_dataset.py --task query --input_path "/path/to/dataset/1.질문/소화기질환" --output_path "/path/to/preprocessed_query_dataset" --n 5 # n : sampling할 query 개수
python preprocess_dataset.py --task corpus --input_path "/path/to/dataset/2.답변/소화기질환" --output_path "/path/to/preprocessed_corpus_dataset"
```

### 3. RAG+ChatGPT
SYSTEM_PROMPT와 USER_PROMPT를 입력하고 아래 예시와 같이 실행 (run.sh 파일 참고)
```bash
export SYSTEM_PROMPT="당신은 의학 전문가 AI이다. 당신은 환자의 질문이나 요청에 대해 상세하고 전문적인 답변을 이해하기 쉽게 제공한다. 그러나, 당신은 의사가 아니기 때문에 직접적인 진료나 진단을 수행할 수는 없다. 이러한 경우, 당신은 환자를 의사의 진료를 받을 수 있도록 안내해야 한다."
export USER_PROMPT="[관련 정보]를 참고하여 [질문]에 답변하세요."
```

3.1. **BM25**
```bash
python run_retrieval.py bm25 --corpus_path "/path/to/preprocessed_query_dataset" \
                            --query_path "/path/to/preprocessed_corpus_dataset" \
                            --k 5 \
                            --chatgpt_model gpt-4o \
                            --system_prompt "$SYSTEM_PROMPT" \
                            --user_prompt "$USER_PROMPT" \
                            --api_key "your-api-key"
```
* corpus_path : 2.에서 생성한 preprocessed corpus dataset의 경로
* query_path : 2.에서 생성한 preprocessed query dataset의 경로
* k : top-k retrieval 결과

3.2. **DPR**
```bash
python run_retrieval.py dpr --corpus_path "/path/to/preprocessed_query_dataset" \
                            --query_path "/path/to/preprocessed_corpus_dataset" \
                            --k 5 \
                            --model_path "klue/bert-base" \
                            --max_length 512 \
                            --chatgpt_model "gpt-4o" \
                            --system_prompt "$SYSTEM_PROMPT" \
                            --user_prompt "$USER_PROMPT" \
                            --api_key "your-api-key"
```
* model_path : huggingface로 불러오는 embedding model 경로 (ex.[klue/bert-base](https://huggingface.co/klue/bert-base))
* max_length : tokenizer max_length

## I/O example
**Input prompt**
```
[관련 정보]

십이지장 궤양의 주요 증상은 식사 후에 나타나는 명치끝의 통증입니다. 통증은 식후에 약 12~72시간 동안 지속됩니다. 
주로 우측 하복부에서 통증이 발생하며, 통증의 위치와 형태는 다양할 수 있습니다. 
초기 증상으로는 오심, 구토, 위부 불쾌감, 상복부 통증, 소화불량 등이 나타날 수 있습니다.

[질문]

십이지장 궤양의 증상에는 어떤 것이 있나요?

[관련 정보]를 참고하여 [질문]에 답변하세요.
```
**ChatGPT response**
```
십이지장 궤양의 주요 증상은 다음과 같습니다:

복통: 특히 상복부 중앙에서 느껴지는 통증으로, 식사 후 몇 시간 뒤에 주로 발생합니다. 공복 시에 통증이 더 심해질 수 있으며, 밤에 통증으로 잠에서 깨는 경우도 있습니다.

속쓰림: 위산이 십이지장을 자극해 속쓰림을 유발할 수 있습니다.

속이 더부룩하거나 가스참: 소화가 잘 되지 않거나 가스가 차는 느낌을 받을 수 있습니다.

구토: 심한 경우 메스꺼움과 구토 증상이 나타날 수 있습니다.

식욕 감소 및 체중 감소: 식사 후 불편함 때문에 식사를 피하게 되어 체중 감소가 나타날 수 있습니다.

흑변: 궤양에서 출혈이 있을 경우 변이 검게 보일 수 있습니다.

궤양이 심해지면 출혈, 천공 등의 합병증이 발생할 수 있으므로, 위와 같은 증상이 지속되면 빠르게 병원 진료를 받는 것이 중요합니다.
```









