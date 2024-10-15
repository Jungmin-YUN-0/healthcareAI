from get_dataset import get_search_target_data, get_queries
from chatgpt import get_chatgpt_multiple_responses
from tqdm import tqdm
import argparse
import json
import datetime
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Run retrieval algorithms (BM25, DPR)")

    parser.add_argument('--query_path', type=str, required=True, help="Path to the query file")    
    parser.add_argument('--chatgpt_model', type=str, default='gpt-4o', help='Name of ChatGPT model name')
    parser.add_argument('--system_prompt', type=str, help='System prompt of ChatGPT')
    parser.add_argument('--user_prompt', type=str, help='Input prompt or instruction for ChatGPT, will be meraged with query')
    parser.add_argument('--api_key', type=str, help='ChatGPT API key')

    args = parser.parse_args()

    logging.info("Getting dataset")

    args.queries = get_queries(args.query_path)

    logging.info("Getting chatgpt response")

    ret_results = get_chatgpt_multiple_responses(args.queries, args.chatgpt_model, args.system_prompt, args.user_prompt, args.api_key)

    os.makedirs('outputs', exist_ok=True)
    current_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
    fout_path = f'{os.getcwd()}/outputs/chatgpt.{current_time}.jsonl'
    fout = open(fout_path, 'w')
    
    for ret_result in ret_results:
        print(json.dumps(ret_result, ensure_ascii=False), file=fout)

    logging.info(f"Process finished, output file saved at : {fout_path}")

if __name__ == "__main__":
    main()