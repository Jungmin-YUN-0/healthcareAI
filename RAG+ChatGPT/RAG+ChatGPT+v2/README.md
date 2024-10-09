# RAG(Retrieval Augmented Generation) + ChatGPT (Version 2)
- Retireval DB : MSD, KDCA 크롤링 데이터셋

## Requirements
* torch
* transformers
* tqdm
* jsonlines
* rank-bm25
* openai
* kiwipiepy

아래와 같은 스크립트를 통해 필요한 라이브러리를 설치

```bash
pip install torch transformers tqdm jsonlines rank-bm25 kiwipiepy
pip install openai==0.28.0
```

## Usage
* BM25+ChatGPT
```bash
bash 1.run_bm24.sh
```

* DPR+ChatGPT
```bash
bash 2.run_dpr.sh
```
