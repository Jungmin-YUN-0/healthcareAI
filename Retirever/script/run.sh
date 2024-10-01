# 구동 순서 예시이고, input, output file path는 직접 수정할 필요가 있습니다.

python preprocess_dataset.py

python run_retrieval.py make_dpr_embeddings

python run_retrieval.py dpr

python run_retrieval.py bm25

python run_retrieval.py rrf