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

### 3. DPR+ChatGPT
Use the DPR-based retrieval mechanism by running:
```bash
bash run_dpr.sh
```

### 4. Only ChatGPT
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
1) BM25   
- `K`: Define the number of top search results to return. Default is 3.   
2) DPR
- `K`: Define the number of top search results to return. Default is 3.   
- `EMBEDDING_PATH`: Provide the path to the precomputed embedding file for DPR retrieval. If the file is missing, it will be **automatically generated**.
- `MODEL_PATH`: Specify the model path for DPR. For Korean text, models like `KoE5` are recommended.   
- `MAX_LENGTH`: Set the maximum input token length for embedding generation. Default is 512.   

### 3. ChatGPT Settings:
- `CHATGPT_MODEL`: Specify the ChatGPT model name to use (e.g., gpt-4o).   
- `SYSTEM_PROMPT`: Tailor ChatGPT's persona and response style.   
- `USER_PROMPT`: Instruction for user-specific context.   
- `API_KEY`: Add your OpenAI API key.

## I/O Format
- Input : .txt file
- Output : .jsonl file
  ```
  {
    "query": "string",
    "ret_results": [
      {
        "question": "string",
        "answer": "string",
        "url": "string"
      }
    ],
    "chatgpt_input": "string",
    "chatgpt_response": "string"
  }
  ```
  - `query`: The user-provided question.
  - `ret_results`:
    Retrieved results related to the query.   
    - `question`: Related question.
    - `answer`: Answer to the question (Ground truth).
    - `url`: The source of the retrieved information.
  - `chatgpt_input`: Input provided to ChatGPT, combining the query and related information.
  - `chatgpt_response`: The system's final response to the user.


- I/O Example
  - Input :
  ```
  고혈압 망막병증이 심해지면 시력을 완전히 잃을 수도 있나요?
  ```
  - Output :
  ```
  {"query": "고혈압 망막병증이 심해지면 시력을 완전히 잃을 수도 있나요?", "ret_results": [{"question": "고혈압 망막병증이 심해지면 시력을 완전히 잃을 수도 있나요?", "answer": "고혈압과 당뇨병은 대표적인 실명의원인입니다.3단계 이후에도 적절한 혈압 조절과 안과적 치료를 받지 못하면 실명할 수 있습니다.", "url": "https://health.kdca.go.kr/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/gnrlzHealthInfoView.do"}, {"question": "고혈압 망막병증이 한쪽 눈에만 생길 수도 있나요?", "answer": "고혈압은 눈뿐만 아니라, 전신의 혈관에 영향을 미칠 수 있습니다. 따라서, 고혈압 망막병증은 대개 양쪽 눈에 동시에 비슷한 양상으로 나타납니다. 한쪽 눈에만 문제가 생긴 경우에는 다른 질환을 의심해야 합니다.", "url": "https://health.kdca.go.kr/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/gnrlzHealthInfoView.do"}, {"question": "시력저하나 두통 외에 고혈압 망막병증을 의심해야 할 증상이 있나요?", "answer": "고혈압은 눈에만 영향을 주는 것이 아닙니다.전신 증상으로 호흡곤란, 흉통 등 심장 증상이 나타날 수 있고 죽상경화성 뇌혈관질환, 만성콩팥병과 관련된 증상도 동반될 수 있습니다.", "url": "https://health.kdca.go.kr/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/gnrlzHealthInfoView.do"}], "chatgpt_input": "관련 의학 지식:\n고혈압과 당뇨병은 대표적인 실명의원인입니다.3단계 이후에도 적절한 혈압 조절과 안과적 치료를 받지 못하면 실명할 수 있습니다.\n고혈압은 눈뿐만 아니라, 전신의 혈관에 영향을 미칠 수 있습니다. 따라서, 고혈압 망막병증은 대개 양쪽 눈에 동시에 비슷한 양상으로 나타납니다. 한쪽 눈에만 문제가 생긴 경우에는 다른 질환을 의심해야 합니다.\n고혈압은 눈에만 영향을 주는 것이 아닙니다.전신 증상으로 호흡곤란, 흉통 등 심장 증상이 나타날 수 있고 죽상경화성 뇌혈관질환, 만성콩팥병과 관련된 증상도 동반될 수 있습니다.\n\n질문: 고혈압 망막병증이 심해지면 시력을 완전히 잃을 수도 있나요?\n\n위 환자의 질문에 대해서 의학적 지식을 기반으로 간결하게 답변해주세요. 추가로 질문이 필요한 경우에는 한 번에 한 가지 질문을 해주세요.", "chatgpt_response": "네, 고혈압 망막병증이 심해지면 시력을 잃을 수 있습니다. 고혈압이 지속적으로 조절되지 않으면 눈의 혈관에 손상이 가해져 시력이 점차적으로 악화될 수 있습니다. 그래서 적절한 혈압 관리와 정기적인 안과 검진이 중요합니다. 혹시 현재 혈압 관리에 어려움을 겪고 계신가요, 아니면 증상의 변화가 있으신가요?"}
  ```



## Project Structure
```
.
├── dataset/                          # Dataset files (search target, queries)
│   ├── health_dataset_1014_spacing.csv  # Example search target dataset
│   ├── health_dataset_1022_spacing.csv  # Example search target dataset
│   ├── queries.txt                      # Example query file
│   └── stopwords-ko.txt              # Stopwords dataset for tokenizer
│
├── dpr_embeddings/                          
│   └── embedding.ko-sroberta-multitask.512.pt         # embedding file example
│
├── scripts/                          # Shell scripts for execution
│   ├── run_bm25.sh                   # BM25-based retrieval + ChatGPT
│   ├── run_dpr.sh                    # DPR-based retrieval + ChatGPT
│   ├── run_hybrid.sh                 # Hybrid retrieval + ChatGPT
│   └── run_chatgpt.sh                # ChatGPT-only execution
│
├── src/                              # Source code for retrieval and generation
│   ├── run_retrieval.py              # Main script for retrieval (BM25/DPR) + ChatGPT
│   ├── run_chatgpt.py                # Main script for ChatGPT-only execution
│   ├── get_dataset.py                # Dataset loading utilities
│   ├── get_retriever.py              # Retriever loading utilities
│   └── chatgpt.py                    # ChatGPT interaction module
│
└── outputs/                          # Output directory for results
│   ├── bm25.0404_081130.jsonl        # Output file example (BM25-based retrieval + ChatGPT) 
│   ├── chatgpt.0404_083026.jsonl     # Output file example (ChatGPT-only execution)
│   ├── chatgpt.0404_083026.jsonl     # Output file example (ChatGPT-only execution)
│   └── hybrid.0404_075803.jsonl      # Output file example (DPR-based retrieval + ChatGPT)
│
└── README.md
```
