import jsonlines

def medical_qa_dataset_corpus(file_path, tokenize=True, return_info=False):

    corpus = []
    infos = []
    with jsonlines.open(file_path) as f:
        for line in f.iter():
            text = line['text']
            if tokenize:
                corpus.append(text.split())
            else:
                corpus.append(text)
            if return_info:
                infos.append(line)
    
    if return_info:
        return corpus, infos
    else:
        return corpus
    
def medical_qa_dataset_qeury(file_path, tokenize=True, return_info=False, return_ground_truths=True):

    queries = []
    infos = []
    ground_truths = []
    with jsonlines.open(file_path) as f:
        for line in f.iter():
            query = line['query']
            if tokenize:
                queries.append(query.split())
            else:
                queries.append(query)
            if return_info:
                infos.append(line)
            if return_ground_truths:
                ground_truths.append(line['disease_name']['kor'])

    if return_info and ground_truths:
        return queries, infos, ground_truths
    elif return_info :
        return queries, infos
    else:
        return queries, ground_truths 