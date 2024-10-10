import jsonlines

def get_corpus(file_path):

    corpus = []
    names = []
    with jsonlines.open(file_path) as f:
        for line in f.iter():
            n = line['name']
            c = line['text']
            corpus.append(c)
            names.append(n)
    
    return corpus, names


def get_queries(file_path):

    queries = []
    names = []
    with jsonlines.open(file_path) as f:
        for line in f.iter():
            n = line['name']
            q = line['text']
            queries.append(q)
            names.append(n)
    
    return queries, names