import pandas as pd

def get_search_target_data(file_path):
    df = pd.read_csv(file_path, encoding='cp949')
    search_target = df['question'].tolist()
    target_metadata = df.to_dict(orient='records')
    return search_target, target_metadata 

def get_queries(file_path):
    queries = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            queries.append(line.strip())
    return queries