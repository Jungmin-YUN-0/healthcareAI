import os
import json
from tqdm import tqdm
import random

random.seed(42)

'''
- script를 실행하기 전에 AI 허브의 "초거대 AI 헬스케어 질의응답 데이터" 다운로드 후 unzip, file_path 재설정 필요
- https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71762
'''


def get_all_json_files(folder_path):
    json_data_list = []
    
    # 주어진 폴더 내의 모든 하위 폴더를 탐색
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 파일 확장자가 .json인 경우
            if file.endswith('.json'):
                # JSON 파일의 전체 경로 생성
                file_path = os.path.join(root, file)
                
                # JSON 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    json_data_list.append(data)
    
    return json_data_list


def preprocess_raw_medical_qa_query():

    output = get_all_json_files('/home/jinhee/IIPL_PROJECT/rapa/dataset/medical_qa/1.질문/소화기질환')
    n = 150
    fout = open(f'/home/jinhee/IIPL_PROJECT/rapa/dataset/preprocessed_dataset/medical_qa_query.소화기질환.{n}.jsonl', 'w')
    random_samples = random.sample(output, n)

    for sample in tqdm(random_samples, desc="preprocess_medical_qa"):
        info = {'disease_name':sample['disease_name'], 'intention':sample['intention'], 'query':sample['question']}
        print(json.dumps(info, ensure_ascii=False), file=fout)
        
    fout.close()


def preprocess_raw_medical_qa_corpus():

    output = get_all_json_files('/home/jinhee/IIPL_PROJECT/rapa/dataset/medical_qa/2.답변/소화기질환')

    fout = open('/home/jinhee/IIPL_PROJECT/rapa/dataset/preprocessed_dataset/medical_qa_corpus.소화기질환.jsonl', 'w')

    corpus = []
    for sample in tqdm(output, desc="preprocess_medical_qa"):
        text = sample['answer']['body']+' '+sample['answer']['conclusion'] # intro는 배고 body랑 conclusion만 포함시킴 (이따금 질문이 겹치기도 해서)
        info = {'disease_name':sample['disease_name'], 'intention':sample['intention'], 'text':text}
        corpus.append(text)
        print(json.dumps(info, ensure_ascii=False), file=fout)
    fout.close()


if __name__=="__main__":
    preprocess_raw_medical_qa_query()
    preprocess_raw_medical_qa_corpus()