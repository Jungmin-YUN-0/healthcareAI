import os
import json
from tqdm import tqdm
import random
import argparse

random.seed(42)

def get_all_json_files(folder_path):
    json_data_list = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    json_data_list.append(data)
    
    return json_data_list

def preprocess_raw_medical_qa_query(input_path, output_path, n):
    output = get_all_json_files(input_path)
    fout = open(output_path, 'w')
    random_samples = random.sample(output, n)

    for sample in tqdm(random_samples, desc="preprocess_medical_qa_query"):
        info = {'disease_name': sample['disease_name'], 'intention': sample['intention'], 'query': sample['question']}
        print(json.dumps(info, ensure_ascii=False), file=fout)
    
    fout.close()

def preprocess_raw_medical_qa_corpus(input_path, output_path):
    output = get_all_json_files(input_path)
    fout = open(output_path, 'w')

    for sample in tqdm(output, desc="preprocess_medical_qa_corpus"):
        text = sample['answer']['body'] + ' ' + sample['answer']['conclusion'] # intro는 제외하고 body와 conclusion만 포함
        info = {'disease_name': sample['disease_name'], 'intention': sample['intention'], 'text': text}
        print(json.dumps(info, ensure_ascii=False), file=fout)
    
    fout.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical QA Preprocessing Script")

    parser.add_argument('--task', choices=['query', 'corpus'], required=True, help="Choose between 'query' or 'corpus' for preprocessing")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input folder containing JSON files")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument('--n', type=int, default=150, help="Number of random samples for query preprocessing (only used for 'query' task)")

    args = parser.parse_args()

    if args.task == 'query':
        preprocess_raw_medical_qa_query(args.input_path, args.output_path, args.n)
    elif args.task == 'corpus':
        preprocess_raw_medical_qa_corpus(args.input_path, args.output_path)
