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
