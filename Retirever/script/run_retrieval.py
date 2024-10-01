from get_dataset import preprocess_medical_qa_dataset_corpus, preprocess_medical_qa_dataset_qeury
from transformers import AutoModel, AutoTokenizer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import torch.nn.functional as F
import torch
import json
import sys
import datetime



### BM25

def bm25(corpus, q, top_n):
    q = q.split()
    retriever = BM25Okapi(corpus)
    ret_results = retriever.get_top_n(q, corpus, top_n)
    sen_ret_results = [' '.join(i) for i in ret_results] 

    return sen_ret_results


def run_bm25(corpus_path, query_path, k=5):
    current_time = datetime.datetime.now().strftime('%m%d_%H%M%S')

    corpus, c_infos = preprocess_medical_qa_dataset_corpus(corpus_path)
    queries, q_infos = preprocess_medical_qa_dataset_qeury(query_path, tokenize=False)

    fout = open(f'/home/jinhee/IIPL_PROJECT/rapa/output/bm25.ret.output.{current_time}.jsonl', 'w')

    for i, q in enumerate(tqdm(queries)):
        ret_res = bm25(corpus, q, k)
        res_dict = {'query':q, 'infos':q_infos[i], 'ret_result':ret_res}
        print(json.dumps(res_dict, ensure_ascii=False), file=fout)

    fout.close()


### DPR

def make_dpr_embeddings(model_path='klue/bert-base', corpus_path='/home/jinhee/IIPL_PROJECT/rapa/dataset/preprocessed_dataset/medical_qa_corpus.jsonl', max_length=256):

    corpus, infos = preprocess_medical_qa_dataset_corpus(corpus_path, tokenize=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path) # madatnlp/km-bert
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    ### make_embedding
    tkd_dataset = []
    max_length = 256

    for sample in tqdm(corpus):
        tkd_sample = tokenizer(sample, padding='max_length', truncation=True, max_length=int(max_length), return_tensors='pt')
        tkd_dataset.append(tkd_sample)

    embeddings = []

    with torch.no_grad():
        for sample in tqdm(tkd_dataset):
            output = model(input_ids=sample['input_ids'].to(device),
                        attention_mask=sample['attention_mask'].to(device), 
                        token_type_ids=sample['token_type_ids'].to(device))
            embeddings.append(output.pooler_output)
        embeddings = torch.cat(embeddings, dim=0)

    print("made dpr embeddings!")

    torch.save(embeddings, f'/home/jinhee/IIPL_PROJECT/rapa/scripts/embeddings.{max_length}.pt')


def run_dpr(corpus_path, query_path, embedding_path, k=5, max_length=256, model_path='klue/bert-base'):

    current_time = datetime.datetime.now().strftime('%m%d_%H%M%S')

    corpus, c_infos = preprocess_medical_qa_dataset_corpus(corpus_path, tokenize=False)
    queries, q_infos = preprocess_medical_qa_dataset_qeury(query_path, tokenize=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    embeddings = torch.load(embedding_path)

    tkd_q_dataset = []
    q_embedding = []
    

    for sample in tqdm(queries):
        tkd_q_sample = tokenizer(sample, padding='max_length', truncation=True, max_length=int(max_length), return_tensors='pt')
        tkd_q_dataset.append(tkd_q_sample)

    with torch.no_grad():
        for sample in tkd_q_dataset:
            output = model(input_ids=sample['input_ids'].to(device),
                        attention_mask=sample['attention_mask'].to(device), 
                        token_type_ids=sample['token_type_ids'].to(device))
            q_embedding.append(output.pooler_output)

    dpr_results = []

    for q_emb in tqdm(q_embedding):
        q_emb_exp = q_emb.expand(embeddings.size(0), -1)
        similarity_scores = F.cosine_similarity(q_emb_exp, embeddings, dim=1)
        rank = torch.argsort(similarity_scores, descending=True)
        dpr_result = []
        for idx in rank:
            dpr_result.append(corpus[idx.item()])
        dpr_results.append(dpr_result)

    fout = open(f'/home/jinhee/IIPL_PROJECT/rapa/output/dpr.ret.output.{current_time}.jsonl', 'w')

    for i, (q, r) in enumerate(zip(queries, dpr_results)):
        output_dict = {'query':q, 'infos':q_infos[i], 'ret_result':r[:k]} # query 순서대로 infos도 인덱싱을 통해 같이 들어감
        print(json.dumps(output_dict, ensure_ascii=False), file=fout)

    fout.close()

### rrf

def rrf_rank(dpr_results, bm25_results, k=10):
    rrf_scores = {}

    for rank, doc in enumerate(dpr_results):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0
        rrf_scores[doc] += 1 / (k + rank + 1)

    for rank, doc in enumerate(bm25_results):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0
        rrf_scores[doc] += 1 / (k + rank + 1)

    sorted_rrf_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    top_k_rrf = [doc for doc, score in sorted_rrf_scores]

    return top_k_rrf


import jsonlines


def run_rrf(bm25_output, dpr_output, k=10):

    current_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
    fout = open(f'/home/jinhee/IIPL_PROJECT/rapa/output/rrf.ret.output.{current_time}.jsonl', 'w')

    bm25_ret_res = []
    dpr_ret_res = []

    with jsonlines.open(bm25_output) as f:
        for line in f.iter():
            bm25_ret_res.append(line)


    with jsonlines.open(dpr_output) as f:
        for line in f.iter():
            dpr_ret_res.append(line)

    for b, d in zip(bm25_ret_res, dpr_ret_res, k):
        query = b['query']
        q_infos = b['infos']
        rrf_result = rrf_rank(b['ret_result'], d['ret_result'])
        output_dict = {'query':query, 'infos':q_infos, 'ret_result':rrf_result[:10]}
        print(json.dumps(output_dict, ensure_ascii=False), file=fout)

    fout.close()


if __name__ == "__main__":
    
    corpus_path = '/home/jinhee/IIPL_PROJECT/rapa/dataset/preprocessed_dataset/medical_qa_corpus.merged.jsonl'
    query_path = '/home/jinhee/IIPL_PROJECT/rapa/dataset/preprocessed_dataset/medical_qa_query.merged.300.jsonl'
    # model_path = 'klue/bert-base'
    model_path = 'madatnlp/km-bert'
    # embedding_path = '/home/jinhee/IIPL_PROJECT/rapa/scripts/embeddings.kluebert.256.pt'
    embedding_path = '/home/jinhee/IIPL_PROJECT/rapa/scripts/embeddings.kmbert.expanded_corpus.256.pt'
    k=0
    max_length=256

    if sys.argv[1] == 'dpr':
        run_dpr(corpus_path=corpus_path, query_path=query_path, embedding_path=embedding_path, k=k, max_length=max_length, model_path=model_path)
    elif sys.argv[1] == 'bm25':
        run_bm25(corpus_path, query_path, k=k)
    elif sys.argv[1] == 'make_dpr_embeddings': 
        # dpr 전에 dpr embeddings를 먼저 생성할 것
        make_dpr_embeddings(model_path=model_path, corpus_path=corpus_path)
    elif sys.argv[1] == 'rrf':
        # rrf 전에 dpr, bm25 output을 먼저 생성할 것
        bm25_output = '/home/jinhee/IIPL_PROJECT/rapa/output/bm25.ret.output.0821_222450.jsonl'
        dpr_output = '/home/jinhee/IIPL_PROJECT/rapa/output/dpr.ret.output.0821_223136.jsonl'
        run_rrf(bm25_output, dpr_output, k=k)