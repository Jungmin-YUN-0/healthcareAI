import jsonlines

def preprocess_medical_qa_dataset_corpus(file_path, tokenize=True, return_info=True):

    print('[preprocess_medical_qa_dataset_corpus]: tokenize=', tokenize)
    print('[preprocess_medical_qa_dataset_corpus]: return_info=', return_info)

    corpus = []
    infos = []
    with jsonlines.open(file_path) as f:
        for line in f.iter():
            disease_name = line['disease_name']
            intention = line['intention']
            text = line['text']
            if tokenize:
                corpus.append(text.split())
            else:
                corpus.append(text)
            if return_info:
                infos.append(line)
    
    return corpus, infos
    
def preprocess_medical_qa_dataset_qeury(file_path, tokenize=True, return_info=True):

    print('[preprocess_medical_qa_dataset_qeury]: tokenize=', tokenize)
    print('[preprocess_medical_qa_dataset_qeury]: return_info=', return_info)

    queries = []
    infos = []
    with jsonlines.open(file_path) as f:
        for line in f.iter():
            disease_name = line['disease_name']
            intention = line['intention']
            query = line['query']
            if tokenize:
                queries.append(query.split())
            else:
                queries.append(query)
            if return_info:
                infos.append(line)
    
    return queries, infos