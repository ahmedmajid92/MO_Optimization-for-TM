import numpy as np
import pandas as pd
import re
import time
import pickle
from itertools import combinations
from gensim.models import CoherenceModel, LdaModel
from gensim.corpora.dictionary import Dictionary

######################################################
## Read The Corpus
######################################################

df = pd.read_csv('Datasets/corpus.csv')
corpus = df.values.tolist()
corpus = [x[0] for x in corpus]

######################################################
## Load the Topics
######################################################

with open("Topics", "rb") as fp:
  topics = pickle.load(fp)

if __name__ ==  '__main__':

    ######################################################
    ## Compute The Topic Coherence
    ######################################################
        
    texts = corpus

    tokenizer = lambda s: re.findall('\w+', s.lower())
    texts = [tokenizer(t) for t in texts]

    dictionary = Dictionary(texts)

    cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')

    start_time = time.time()

    coherence_per_topic = cm.get_coherence_per_topic()
    #print(coherence_per_topic)

    ind = coherence_per_topic.index(min(coherence_per_topic))
    del topics[ind]
    del coherence_per_topic[ind]

    print("--- %s seconds ---" % (time.time() - start_time))

    with open("Topics", "wb") as fp:
       pickle.dump(topics, fp)