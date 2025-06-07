from get_response import get_RAG_chatgpt_multiple_responses, get_openllm_generations
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import torch.nn.functional as F
import torch
import argparse
import json
import datetime
import logging
import os


from get_dataset import get_corpus, get_queries
from get_retriever import run_bm25, run_dpr, run_rrf
from hyde import HydeSystem
from kcomp import run_kcomp
import argparse
import json
import datetime
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run retrieval algorithms (BM25, DPR)")

    parser.add_argument('mode', choices=['bm25', 'dpr', 'hybrid', 'hyde', 'kcomp', 'hyde_kcomp'], help="Which mode to run: bm25, dpr, hybrid")
    parser.add_argument('--search_target_path', type=str, required=True, help="Path to the search_target file")
    parser.add_argument('--query_path', type=str, required=True, help="Path to the query file")    
    parser.add_argument('--corpus_embedding_path', type=str, help="Path to the DPR embeddings file (required for dpr)")
    parser.add_argument('--embedding_model_name', type=str, default='klue/bert-base', help="Path to the model (for DPR)")
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=256, help="Maximum token length for embeddings")
    parser.add_argument('--k', type=int, default=5, help="Number of top results to return")
    parser.add_argument('--n', type=int, default=5, help="Number of response num for hyde system")
    parser.add_argument('--chatgpt_model_name', type=str, default='gpt-4o', help='Name of ChatGPT model name')
    parser.add_argument('--llm_model_name', type=str, default='yanolja/EEVE-Korean-Instruct-10.8B-v1.0', help='Name of OpenLLM model name')
    parser.add_argument('--system_prompt', type=str, help='System prompt of ChatGPT')
    parser.add_argument('--user_prompt', type=str, help='Input prompt or instruction for ChatGPT, will be meraged with query')
    parser.add_argument('--api_key', type=str, help='ChatGPT API key')
    parser.add_argument('--generation_method', type=str, default='openllm', choices=['openllm', 'apillm'], help='Method to generate responses from ChatGPT')

    args = parser.parse_args()
    logger.info("Getting dataset")

    corpus, _ = get_corpus(args.search_target_path)
    queries = get_queries(args.query_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading LLM model: {args.llm_model_name}")
    
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    llm_model.eval()

    logger.info(f"Loading embedding model: {args.embedding_model_name}")

    emb_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name)
    emb_model = AutoModel.from_pretrained(args.embedding_model_name).to(device)
    emb_model.eval()

    if args.mode == 'bm25':
        logger.info("Starting BM25 retrieval")
        ret_results = run_bm25(logger, corpus, queries, args.k)

    elif args.mode == 'dpr':
        logger.info("Starting DPR retrieval")
        ret_results = run_dpr(logger, corpus, queries, args.k, args.embedding_model_name, args.corpus_embedding_path, args.max_length)

    elif args.mode == 'hybrid':
        logger.info("Starting Hybrid(rrf) retrieval")
        ret_results_bm25 = run_bm25(logger, corpus, queries, args.k)
        ret_results_dpr = run_dpr(logger, corpus, queries, args.k, args.embedding_model_name, args.corpus_embedding_path, args.max_length)
        ret_results = run_rrf(ret_results_bm25, ret_results_dpr, args.k)
    
    elif args.mode == 'hyde':
        logger.info("Starting hyde retrieval")
        hyde = HydeSystem(llm_model, llm_tokenizer, emb_model, emb_tokenizer, args.max_length, args.max_new_tokens, corpus, args.corpus_embedding_path)
        ret_results = hyde.run_hyde(queries, args.n, args.k)
    
    elif args.mode == 'kcomp':
        logger.info("Starting kcomp retrieval")
        ret_results = run_dpr(logger, corpus, queries, args.k, args.embedding_model_name, args.corpus_embedding_path, args.max_length)
        ret_results = run_kcomp(ret_results, llm_model, llm_tokenizer, device)

    elif args.mode == 'hyde_kcomp':
        hyde = HydeSystem(llm_model, llm_tokenizer, emb_model, emb_tokenizer, args.max_length, args.max_new_tokens, corpus, args.corpus_embedding_path)
        ret_results = hyde.run_hyde(queries, args.n, args.k)
        ret_results = run_kcomp(ret_results, llm_model, llm_tokenizer, device)


    logger.info("Retrieval finished")

    if args.generation_method == 'apillm':
        logger.info("Getting API generations")
        ret_results = get_RAG_chatgpt_multiple_responses(ret_results, args.chatgpt_model_name, args.system_prompt, args.user_prompt, args.api_key)
    elif args.generation_method == 'openllm':
        logger.info("Getting OpenLLM generations")
        generation_mode = 'kcomp' if args.mode in ['kcomp', 'hyde_kcomp'] else 'default'
        ret_results = get_openllm_generations(ret_results, llm_model, llm_tokenizer, generation_mode)


    current_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
    model_nickname = args.llm_model_name.split('/')[-1]

    fout_path = os.path.join(os.path.dirname(os.getcwd()), 'outputs', f'{args.mode}.{model_nickname}.{current_time}.jsonl')
    fout = open(fout_path, 'w')
    
    for ret_result in ret_results:
        print(json.dumps(ret_result, ensure_ascii=False), file=fout)

    fout.close()

    logging.info(f"Process finished, output file saved at : {fout_path}")

if __name__ == "__main__":
    main()