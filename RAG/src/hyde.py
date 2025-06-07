import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from get_retriever import make_dpr_embedding
import torch.nn.functional as F
from typing import List, Dict, Tuple
import json
import logging
from tqdm import tqdm
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HydeSystem:
    def __init__(self, 
                 llm_model,
                 llm_tokenizer,
                 emb_model,
                 emb_tokneizer,
                 max_length: int = 512,
                 max_new_tokens: int = 500,
                 corpus = None,
                 corpus_embedding_path: str = None,):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.corpus = corpus
        self.corpus_embedding_path = corpus_embedding_path
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.emb_model = emb_model
        self.emb_tokenizer = emb_tokneizer
            
    def generate_multiple_responses(self, query: str, num_responses: int = 5) -> List[str]:
        logger.info(f"Generating {num_responses} responses for query")
        
        prompt_template = """다음 의료 관련 [질문]에 대해 정확하고 유용한 [답변]을 제공해주세요. [답변]은 한번만 제공해주세요.
        
        [질문]
        {query}
        
        [답변]
        """
        responses = []
        prompt_filled = prompt_template.format(query=query)

        inputs = self.llm_tokenizer(prompt_filled, return_tensors="pt").to(self.device)

        for i in tqdm(range(num_responses), desc="generating..."):
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                        
            generated = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.llm_tokenizer.decode(generated, skip_special_tokens=True).strip()

            if response:
                responses.append(response)

        logger.info(f"Generations {responses}")
        return responses
    
    def get_response_embeddings(self, responses: List[str]) -> np.ndarray:

        logger.info("Computing embeddings for generations")
        
        tokenized_resp = []
        embeddings = []

        for sample in tqdm(responses):
            tkd_q_sample = self.emb_tokenizer(sample, padding='max_length', truncation=True, max_length=int(self.max_length), return_tensors='pt')
            tokenized_resp.append(tkd_q_sample)

        with torch.no_grad():
            for sample in tokenized_resp:
                output = self.emb_model(input_ids=sample['input_ids'].to(self.device),
                            attention_mask=sample['attention_mask'].to(self.device))
                embeddings.append(output.pooler_output.cpu())
            
        stacked_embeddings = torch.cat(embeddings, dim=0)
        mean_embedding = torch.mean(stacked_embeddings, dim=0)

        return mean_embedding

    def generate_all_responses_embeddings(self, queries: List[str], num_responses: int = 5) -> List[List[str]]:
        all_responses = []
        all_embeddings = []
        
        for query in queries:
            responses = self.generate_multiple_responses(query, num_responses)
            all_responses.append(responses)

        for responses in all_responses:
            embdding = self.get_response_embeddings(responses)
            all_embeddings.append(embdding)
        
        return all_responses, all_embeddings
    
    def get_corpus_embeddings(self):
        if self.corpus_embedding_path == None:
            logger.info("There's no corpus embedding_path for DPR retreival, Creating new embedding")

            os.makedirs(f'{os.path.dirname(os.getcwd())}/dpr_embeddings', exist_ok=True)
            emb_model_name = self.emb_model.config.name_or_path
            model_nickname = emb_model_name.split('/')[-1]
            self.corpus_embedding_path = f'{os.path.dirname(os.getcwd())}/dpr_embeddings/embedding.{model_nickname}.{self.max_length}.pt'
            make_dpr_embedding(logger, emb_model_name, self.corpus, self.max_length, self.corpus_embedding_path)

        return torch.load(self.corpus_embedding_path).cpu()


    def retrieve_passages(self, query_embeddings, queries, top_k, responses):
        
        logger.info(f"Retrieving top-{top_k} similar passages using DPR method")

        corpus_embeddngs = self.get_corpus_embeddings()


        dpr_results = []
        for query_embedding in tqdm(query_embeddings, desc='running dpr retrieval'):
            expanded_query_embedding = query_embedding.expand(corpus_embeddngs.size(0), -1)
            similarity_scores = F.cosine_similarity(expanded_query_embedding, corpus_embeddngs, dim=1)
            rank = torch.argsort(similarity_scores, descending=True)
            dpr_result = [self.corpus[idx.item()] for idx in rank][:top_k]
            dpr_results.append(dpr_result)
        
        ret_results = []

        for i, (q, r) in enumerate(zip(queries, dpr_results)):
            ret_results.append({'query': q, 'ret_results': r, 'hyde_llm_generations': responses})

        logger.info(f"Retrieval results: {ret_results}")
        logger.info(f"Completed pipeline")

        return ret_results
    
    
    def run_hyde(self, queries: str, num_responses: int = 5, top_k: int = 3) -> Dict:
        logger.info(f"Starting hyde pipeline")

        responses, responses_embedding = self.generate_all_responses_embeddings(queries, num_responses)
        retrieved_passages = self.retrieve_passages(responses_embedding, queries, top_k, responses)

        return retrieved_passages

