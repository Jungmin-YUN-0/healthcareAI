# Retriever
- Supported retrieval method : BM25, DPR, RRF
- Retrieval DB : MSD, KDCA 크롤링 데이터셋

## Requirements
* torch
* transformers
* tqdm
* jsonlines
* rank-bm25
* kiwipiepy

아래와 같은 스크립트를 통해 필요한 라이브러리를 설치

```bash
pip install torch transformers tqdm jsonlines rank-bm25 kiwipiepy
```

## Usage
**데이터셋이나 DPR 모델을 변경할 경우, Embedding을 새로 생성하여야합니다. (embedding path를 입력 받지 않을 경우 자동 생성)**
* BM25
```bash
bash 1.run_bm25.sh
```

* DPR
```bash
bash 2.run_dpr.sh
```

* RRF
```bash
bash 3.run_rrf.sh
```
