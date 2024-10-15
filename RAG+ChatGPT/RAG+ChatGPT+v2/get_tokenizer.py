from kiwipiepy import Kiwi, Match
from kiwipiepy.utils import Stopwords
from konlpy.tag import Okt
import os

def kiwi_tokenizer(doc):
    kiwi = Kiwi()
    stopwords = Stopwords()
    tokens = kiwi.tokenize(doc, normalize_coda=True, stopwords=stopwords)
    form_list = [token.form for token in tokens] 
    return form_list 


def okt_tokenizer(doc):
        okt = Okt()
        stopwords_path = os.getcwd()+'/dataset/stopwords-ko.txt'
        with open(stopwords_path, 'r', encoding='utf-8') as file:
            stopwords = file.read().splitlines()
        okt_result = okt.morphs(doc)
        tokens = [t for t in okt_result if t not in stopwords]
        return tokens

def get_tokenizer(tokenizer_name):
     if tokenizer_name == 'kiwi':
        return kiwi_tokenizer
     elif tokenizer_name == 'okt':
        return okt_tokenizer