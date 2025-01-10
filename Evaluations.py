import numpy as np
import pandas as pd
import re
import pickle
from itertools import combinations
from gensim.models import CoherenceModel, LdaModel
from gensim.corpora.dictionary import Dictionary

######################################################
## Read The Corpus
######################################################

df = pd.read_csv('C:/Users/Predator/Desktop/project/Topic_Modelling_Project/Datasets/corpus.csv')
corpus = df.values.tolist()
corpus = [x[0] for x in corpus]

######################################################
## Load the Topics
######################################################

with open("Topics", "rb") as fp:
  topics = pickle.load(fp)

######################################################
## Functions used in Evaluations
######################################################

def pairwise_jaccard_diversity(topics, topk=10):

    dist = 0
    count = 0
    for list1, list2 in combinations(topics, 2):
        js = 1 - len(set(list1).intersection(set(list2))) / len(set(list1).union(set(list2)))
        dist = dist + js
        count = count + 1
    return dist / count


def proportion_unique_words(topics, topk=10):

    if topk > len(topics[0]):
        raise Exception('Words in topics are less than '+str(topk))
    else:
        unique_words = set()
        for topic in topics:
            unique_words = unique_words.union(set(topic[:topk]))
        puw = len(unique_words) / (topk * len(topics))
        return puw


if __name__ ==  '__main__':

    ######################################################
    ## Compute The Topic Coherence
    ######################################################
        
    texts = corpus

    tokenizer = lambda s: re.findall('\w+', s.lower())
    texts = [tokenizer(t) for t in texts]

    dictionary = Dictionary(texts)

    cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')

    TC = cm.get_coherence()
    print("Topic Coherence = ", TC)

    ######################################################
    ## compute (Diversity) average pairwise jaccard
    ## distance Evaluation and Propotion Unique Words
    ######################################################

    TDJ = pairwise_jaccard_diversity(topics, topk=10)
    print("Topic Diversity (Jaccard) =", TDJ)
    TDP = proportion_unique_words(topics, topk=3)
    print("Topic Diversity (Propotion) =", TDP)

    TQ = TC * TDJ
    print("TQ =", TQ)

    #coherence_per_topic = cm.get_coherence_per_topic()

    #print("coherence_per_topic for u_mass:")
    #print(coherence_per_topic)