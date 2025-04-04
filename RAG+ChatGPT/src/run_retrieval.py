from chatgpt import get_RAG_chatgpt_multiple_responses
from transformers import AutoModel, AutoTokenizer
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
import argparse
import json
import datetime
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run retrieval algorithms (BM25, DPR)")

    parser.add_argument('mode', choices=['bm25', 'dpr', 'hybrid'], help="Which mode to run: bm25, dpr, hybrid")
    parser.add_argument('--search_target_path', type=str, required=True, help="Path to the search_target file")
    parser.add_argument('--query_path', type=str, required=True, help="Path to the query file")    
    parser.add_argument('--embedding_path', type=str, help="Path to the DPR embeddings file (required for dpr)")
    parser.add_argument('--model_path', type=str, default='klue/bert-base', help="Path to the model (for DPR)")
    parser.add_argument('--max_length', type=int, default=256, help="Maximum token length for embeddings")
    parser.add_argument('--k', type=int, default=5, help="Number of top results to return")
    parser.add_argument('--chatgpt_model', type=str, default='gpt-4o', help='Name of ChatGPT model name')
    parser.add_argument('--system_prompt', type=str, help='System prompt of ChatGPT')
    parser.add_argument('--user_prompt', type=str, help='Input prompt or instruction for ChatGPT, will be meraged with query')
    parser.add_argument('--api_key', type=str, help='ChatGPT API key')

    args = parser.parse_args()
    logger.info("Getting dataset")

    corpus, _ = get_corpus(args.search_target_path)
    queries = get_queries(args.query_path)

    if args.mode == 'bm25':
        logger.info("Starting BM25 retrieval")
        ret_results = run_bm25(logger, corpus, queries, args.k)

    elif args.mode == 'dpr':
        logger.info("Starting DPR retrieval")
        ret_results = run_dpr(logger, corpus, queries, args.k, args.model_path, args.embedding_path, args.max_length)

    elif args.mode == 'hybrid':
        logger.info("Starting Hybrid(rrf) retrieval")
        ret_results_bm25 = run_bm25(logger, corpus, queries, args.k)
        ret_results_dpr = run_dpr(logger, corpus, queries, args.k, args.model_path, args.embedding_path, args.max_length)
        ret_results = run_rrf(ret_results_bm25, ret_results_dpr, args.k)

    logger.info("Retrieval finished")
    logger.info("Getting chatgpt response")

    ret_results = get_RAG_chatgpt_multiple_responses(ret_results, args.chatgpt_model, args.system_prompt, args.user_prompt, args.api_key)

    os.makedirs(os.path.join(os.path.dirname(os.getcwd()), 'outputs'), exist_ok=True)
    current_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
    fout_path = os.path.join(os.path.dirname(os.getcwd()), 'outputs', f'{args.mode}.{current_time}.jsonl')
    fout = open(fout_path, 'w')
    
    for ret_result in ret_results:
        print(json.dumps(ret_result, ensure_ascii=False), file=fout)

    logging.info(f"Process finished, output file saved at : {fout_path}")

if __name__ == "__main__":
    main()