# RAG(Retrieval Augmented Generation) + ChatGPT (Version 2)
This project implements a Retrieval-Augmented Generation (RAG) system using BM25 and Dense Passage Retrieval (DPR) methods to enhance responses from ChatGPT [(Reference paper)](https://arxiv.org/pdf/2302.00083). The tool is specifically designed for healthcare-related queries, leveraging an intelligent retrieval mechanism combined with ChatGPT's generative capabilities to provide accurate, concise, and contextually relevant answers.

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
pip install openai==0.28.0
```

## Usage
Before running the scripts, ensure the following configurations are updated in the .sh files:

### 1. BM25+ChatGPT
Run the following command to use BM25 for retrieval and ChatGPT for response generation:
```bash
bash run_bm25.sh
```

### 2. DPR+ChatGPT
Use the DPR-based retrieval mechanism by running:
```bash
bash run_dpr.sh
```

### 3. Only ChatGPT
If retrieval is not required, directly invoke ChatGPT:
```bash
bash run_chatgpt.sh
```

## Configuration
This section provides details about the key configurations required to run the script (.sh files). Customize each component as needed to fit your specific use case.

### 1. File Paths:
- `SEARCH_TARGET_PATH`: Path to the search target dataset (e.g., path/to/health_dataset_1014_spacing.csv).   
- `QUERY_PATH`: Path to the query file (list of questions).   

### 2. Retriever Settings:
- `K`: Define the number of top search results to return. Default is 3.   
1) BM25   
- `TOKENIZER`: Select tokenizer (`okt` or `kiwi`).
2) DPR   
- `EMBEDDING_PATH`: Provide the path to the precomputed embedding file for DPR retrieval. If the file is missing, it will be **automatically generated**.
- `MODEL_PATH`: Specify the model path for DPR. For Korean text, models like `KoE5` are recommended.   
- `MAX_LENGTH`: Set the maximum input token length for embedding generation. Default is 512.   

### 3. ChatGPT Settings:
- `CHATGPT_MODEL`: Specify the ChatGPT model name to use (e.g., gpt-4o).   
- `SYSTEM_PROMPT`: Tailor ChatGPT's persona and response style.   
- `USER_PROMPT`: Instruction for user-specific context.   
- `API_KEY`: Add your OpenAI API key.

## Project Structure
```
.
├── dataset/                          # Dataset files (search target, queries)
│   ├── health_dataset_1014_spacing.csv  # Example search target dataset
│   ├── health_dataset_1022_spacing.csv  # Example search target dataset
│   ├── queries.txt                      # Example query file
│   ├── stopwords-ko.txt              # Stopwords dataset for tokenizer
│
├── scripts/                          # Shell scripts for execution
│   ├── run_bm25.sh                   # BM25-based retrieval + ChatGPT
│   ├── run_dpr.sh                    # DPR-based retrieval + ChatGPT
│   ├── run_chatgpt.sh                # ChatGPT-only execution
│
├── src/                              # Source code for retrieval and generation
│   ├── run_retrieval.py              # Main script for retrieval (BM25/DPR)
│   ├── get_dataset.py                # Dataset loading utilities
│   ├── get_tokenizer.py              # Tokenizer utilities
│   ├── chatgpt.py                    # ChatGPT interaction module
│
└── outputs/                          # Output directory for results
│
└── README.md
```
