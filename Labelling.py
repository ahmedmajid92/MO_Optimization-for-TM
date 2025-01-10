import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

##############################################################
### Read the Interests of Authors ###
##############################################################

dfa = pd.read_csv('C:/Users/Predator/Desktop/project/Topic_Modelling_Project/Datasets/interests.csv')
dfa = dfa.drop_duplicates()

interests = dfa.values.tolist()
interests = [x[0] for x in interests]

######################################################
## Load the Topics
######################################################

with open("Topics", "rb") as fp:
  topics = pickle.load(fp)

## Merge each topic words into sentence
topics_words = [' '.join(x) for x in topics]


##############################################################
## Use Sbert to encode the interests and topics words
##############################################################

model = SentenceTransformer('all-MiniLM-L6-v2')

interests_embeddings = model.encode(interests)

topics_embeddings = model.encode(topics_words)

##############################################################
## Choose the right label for each topic
##############################################################

labelled_topics = {}
for i in range(len(topics)):
  interests_embeddings = model.encode(interests)
  result = cosine_similarity([topics_embeddings[i]],interests_embeddings)
  ind = result.argmax()
  label = interests[ind]
  labelled_topics[label] = topics[i]
  del interests[ind]

## Print the Labelled Topics
#print(len(labelled_topics))
#print(labelled_topics)

for k, v in labelled_topics.items():
    print(k," is the Label for the topic : ", v)