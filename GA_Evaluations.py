import numpy as np
import pandas as pd
import re
import pickle
from itertools import combinations
from gensim.models import CoherenceModel, LdaModel
from gensim.corpora.dictionary import Dictionary


def proportion_unique_words(topics, topk=10):

    if topk > len(topics[0]):
        raise Exception('Words in topics are less than '+str(topk))
    else:
        unique_words = set()
        for topic in topics:
            unique_words = unique_words.union(set(topic[:topk]))
        puw = len(unique_words) / (topk * len(topics))
        return puw


def EvaluateSol(topics):
    ######################################################
    ## Read The Corpus
    ######################################################

    df = pd.read_csv('C:/Users/Predator/Desktop/project/Topic_Modelling_Project/Datasets/corpus.csv')
    corpus = df.values.tolist()
    corpus = [x[0] for x in corpus]

    ######################################################
    ## Compute The Topic Coherence
    ######################################################
        
    texts = corpus

    tokenizer = lambda s: re.findall('\w+', s.lower())
    texts = [tokenizer(t) for t in texts]

    dictionary = Dictionary(texts)

    cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')

    TC = cm.get_coherence()

    ######################################################
    ## compute (Diversity) average pairwise jaccard
    ## distance Evaluation and Propotion Unique Words
    ######################################################

    TDP = proportion_unique_words(topics, topk=3)

    TQ = TC * TDP

    CN = len(topics)
    return TC, TDP, TQ, CN