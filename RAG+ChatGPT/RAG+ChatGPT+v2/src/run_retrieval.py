from get_dataset import get_search_target_data, get_queries
from get_tokenizer import get_tokenizer
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


### DPR
def make_dpr_embedding(model_path, search_target, max_length, embedding_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    tkd_dataset = []
    for sample in tqdm(search_target):
        tkd_sample = tokenizer(sample, padding='max_length', truncation=True, max_length=int(max_length), return_tensors='pt')
        tkd_dataset.append(tkd_sample)

    embeddings = []
    with torch.no_grad():
        for sample in tqdm(tkd_dataset):
            output = model(input_ids=sample['input_ids'].to(device),
                           attention_mask=sample['attention_mask'].to(device))
                        #    token_type_ids=sample['token_type_ids'].to(device))
            embeddings.append(output.pooler_output)
        embeddings = torch.cat(embeddings, dim=0)    

    torch.save(embeddings, embedding_path)

    logging.info(f"DPR embedding is created at {embedding_path}")


def run_dpr(search_target, queries, target_metadata, k, model_path, embedding_path, max_length):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    embeddings = torch.load(embedding_path)

    tokenized_queries = []
    query_embeddings = []
    for sample in tqdm(queries):
        tkd_q_sample = tokenizer(sample, padding='max_length', truncation=True, max_length=int(max_length), return_tensors='pt')
        tokenized_queries.append(tkd_q_sample)

    with torch.no_grad():
        for sample in tokenized_queries:
            output = model(input_ids=sample['input_ids'].to(device),
                           attention_mask=sample['attention_mask'].to(device))
                        #    token_type_ids=sample['token_type_ids'].to(device))
            query_embeddings.append(output.pooler_output)

    dpr_results = []
    for query_embedding in tqdm(query_embeddings, desc='running dpr retrieval'):
        expanded_query_embedding = query_embedding.expand(embeddings.size(0), -1)
        similarity_scores = F.cosine_similarity(expanded_query_embedding, embeddings, dim=1)
        rank = torch.argsort(similarity_scores, descending=True)
        dpr_result = [{'question':search_target[idx.item()], 'answer':target_metadata[idx.item()]['answer'], 'url':target_metadata[idx.item()]['url']} for idx in rank[:k]]
        dpr_results.append(dpr_result)

    ret_results = []
    
    for i, (q, r) in enumerate(zip(queries, dpr_results)):
        ret_results.append({'query': q, 'ret_results':r[:k]})

    return ret_results


### BM25
def run_bm25(search_target, queries, target_metadata, k, tokenizer_name):
    ret_results = []
    tokenizer = get_tokenizer(tokenizer_name)

    logging.info("Tokenizing search_target")

    tokenized_search_target = []
    for doc in tqdm(search_target, desc='tokenizing search_target'):
        tokenized_search_target.append(tokenizer(doc))

    logging.info("Tokenized search_target")
    # BM25 retrieval
    retriever = BM25Okapi(tokenized_search_target)

    for i, q in enumerate(tqdm(queries, desc='running bm25 retrieval')):

        top_results = retriever.get_top_n(tokenizer(q), list(range(len(search_target))), len(search_target))
        formatted_results = [{'question':search_target[t], 'answer':target_metadata[t]['answer'], 'url':target_metadata[t]['url']} for t in top_results[:k]]                

        ret_results.append({'query': q, 'ret_results': formatted_results})

    return ret_results


def main():
    parser = argparse.ArgumentParser(description="Run retrieval algorithms (BM25, DPR)")

    parser.add_argument('mode', choices=['bm25', 'dpr'], help="Which mode to run: bm25, dpr")
    parser.add_argument('--search_target_path', type=str, required=True, help="Path to the search_target file")
    parser.add_argument('--query_path', type=str, required=True, help="Path to the query file")    
    parser.add_argument('--tokenizer', choices=['okt', 'kiwi'], default='okt', type=str, help="Tokenizer for BM25")
    parser.add_argument('--embedding_path', type=str, help="Path to the DPR embeddings file (required for dpr)")
    parser.add_argument('--model_path', type=str, default='klue/bert-base', help="Path to the model (for DPR)")
    parser.add_argument('--max_length', type=int, default=256, help="Maximum token length for embeddings")
    parser.add_argument('--k', type=int, default=5, help="Number of top results to return")
    parser.add_argument('--chatgpt_model', type=str, default='gpt-4o', help='Name of ChatGPT model name')
    parser.add_argument('--system_prompt', type=str, help='System prompt of ChatGPT')
    parser.add_argument('--user_prompt', type=str, help='Input prompt or instruction for ChatGPT, will be meraged with query')
    parser.add_argument('--api_key', type=str, help='ChatGPT API key')

    args = parser.parse_args()

    logging.info("Getting dataset")

    args.search_target, args.target_metadata = get_search_target_data(args.search_target_path)
    args.queries = get_queries(args.query_path)

    logging.info("Starting retrieval")

    if args.mode == 'bm25':
        ret_results = run_bm25(args.search_target, args.queries, args.target_metadata, args.k, args.tokenizer)

    elif args.mode == 'dpr':
        if args.embedding_path == None:
            logging.info("There's no embedding_path for DPR retreival, Creating new embedding")

            os.makedirs('dpr_embeddings', exist_ok=True)
            model_name = args.model_path.split('/')[-1]
            args.embedding_path = os.path.join(os.path.dirname(os.getcwd()), 'dpr_embeddings', f'embedding.{model_name}.{args.max_length}.pt')
            make_dpr_embedding(args.model_path, args.search_target_path, args.max_length, args.embedding_path)

        ret_results = run_dpr(args.search_target, args.queries, args.target_metadata, args.k, args.model_path, args.embedding_path, args.max_length)
    
    logging.info("Retrieval finished")
    logging.info("Getting chatgpt response")

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