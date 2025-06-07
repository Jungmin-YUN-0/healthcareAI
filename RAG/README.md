# RAG(Retrieval Augmented Generation)
This project implements a Retrieval-Augmented Generation (RAG) system using BM25 and Dense Passage Retrieval (DPR) methods to enhance generations from LLMs [(Reference paper)](https://arxiv.org/pdf/2302.00083). The tool is specifically designed for healthcare-related queries, leveraging an intelligent retrieval mechanism combined with LLMs' generative capabilities to provide accurate, concise, and contextually relevant answers.

## System Requirements
* Python 3.8 or higher.
* `torch`
* `transformers`
* `tqdm`
* `jsonlines`
* `rank-bm25`
* `openai`
* `kiwipiepy`

## Installation
Install the required Python libraries using pip:
```bash
pip install torch transformers tqdm jsonlines rank-bm25 kiwipiepy
```

## Usage
Run the following command to use RAG system. :
```bash
bash run_{retrieval_method}.sh
```
You can choose one of the {retrival_method} form the list below.
* bm25
* dpr
* hybrid (rrf)
* [hyde](https://aclanthology.org/2023.acl-long.99.pdf) 
* [kcomp](https://aclanthology.org/2025.naacl-long.351.pdf)
* hyde_kcomp (mixed)
Before running the scripts, ensure the following configurations are updated in the .sh files

## Configuration
This section provides details about the key configurations required to run the script (.sh files). Customize each component as needed to fit your specific use case.

- `SEARCH_TARGET_PATH`: Path to the search target dataset (e.g., path/to/health_dataset_1014_spacing.csv).   
- `QUERY_PATH`: Path to the query file (list of questions).
- `MAX_LENGTH`: Set the maximum input token length for embedding generation. Default is 512.   
- `K`: Define the number of top search results to return. Default is 5.
- `N`: Define the number of generation for hyde pipeline. Default is 5.
- `EMBEDDING_PATH`: Provide the path to the precomputed embedding file for DPR retrieval. If the file is missing, it will be **automatically generated**.
- `EMBEDDING_MODEL_NAME`: Specify the model path for DPR embedding.
- `LLM_MODELS`: Generation model name.
