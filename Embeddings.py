import numpy as np # linear algebra
import pandas as pd
import umap
import pickle
from sentence_transformers import SentenceTransformer

######################################################
## Read The Sentence List 
######################################################

df = pd.read_csv('Datasets/Sentence_list.csv')
Sentence_list = df.values.tolist()
Sentence_list = [x[0] for x in Sentence_list]

########################################################################
# Apply Sbert using sentence Transformer
########################################################################

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(Sentence_list)

#Print the embeddings
#for sentence, embedding in zip(Sentence_list, embeddings):
  #print("Sentence:", sentence)
  #print("Embedding:", embedding)

########################################################################
# Dimensionality reduction Step (UMAP algorithm)
########################################################################

umap_embeddings = umap.UMAP(n_neighbors=32,
                            n_components=15,
                            metric='cosine').fit_transform(embeddings)
#print("umap_embeddings:", umap_embeddings)

########################################################################
# Save Embeddiing Results for Modelling
########################################################################

with open("embeddings", "wb") as fp:
  pickle.dump(umap_embeddings, fp)