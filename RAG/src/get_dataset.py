import pandas as pd

def get_corpus(file_path):

    df = pd.read_csv(file_path)
    list_of_dicts = df.to_dict(orient='records')
    corpus = list(set(i['Content'] for i in list_of_dicts)) # 중복 요소 제거

    return corpus, list_of_dicts

def get_queries(file_path):
    queries = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            queries.append(line.strip())
    return queries