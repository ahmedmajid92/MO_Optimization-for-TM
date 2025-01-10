import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import re
import time
import hdbscan
import umap

######################################################
## Read The Sentence List 
######################################################

df = pd.read_csv('C:/Users/Predator/Desktop/project/Topic_Modelling_Project/Datasets/Sentence_list.csv')
Sentence_list = df.values.tolist()
Sentence_list = [x[0] for x in Sentence_list]

###########################################################
## Load the Umap Embeddings
###########################################################

with open("embeddings", "rb") as fp:
  embeddings = pickle.load(fp)


start_time = time.time()


###########################################################
# Clustering Step using HDBSCAN Algorithm
###########################################################

cluster = hdbscan.HDBSCAN(min_cluster_size=152,
                          min_samples=86,
                          cluster_selection_epsilon=0.5,
                          metric='euclidean',
                          cluster_selection_method='eom').fit(embeddings)

print("No. of Clusters is : ",cluster.labels_.max() + 1)

########################################################################
# Clusters Repersentation
########################################################################
'''
# Prepare data
umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
result = pd.DataFrame(umap_data, columns=['x', 'y'])
result['labels'] = cluster.labels_

# Visualize clusters
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
plt.colorbar()
plt.show()
'''
########################################################################
# Topic Creation
########################################################################

def c_tf_idf(documents, m):
    count = CountVectorizer().fit(documents)
    t = count.transform(documents).toarray()
    #print("The r is : ", t)
    w = t.sum(axis=1)
    #print("the w is : ",w)
    tf = np.divide(t.T, w)
    #print("the t.T is : ",t.T)
    #print("the tf is : ",tf)
    sum_t = t.sum(axis=0)
    #print("the sum_t is: ",sum_t)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    #print("The idf is : ",idf)
    tf_idf = np.multiply(tf, idf)
    #print("The td_idf: ", tf_idf)

    return tf_idf, count

sent_df = pd.DataFrame(Sentence_list, columns=["Sent"])
sent_df['Topic'] = cluster.labels_
sent_df['Doc_ID'] = range(len(sent_df))
sent_per_topic = sent_df.groupby(['Topic'], as_index = False).agg({'Sent': ' '.join})
sent_per_topic = sent_per_topic.drop(index=0) #ignore the outliers
sent_per_topic.reset_index(inplace = True)

tf_idf, count = c_tf_idf(sent_per_topic.Sent.values, m=len(Sentence_list))

########################################################################
# topic representation
########################################################################

def extract_top_n_words_per_topic(tf_idf, count, sent_per_topic, n=20):
    words = count.get_feature_names_out()
    labels = list(sent_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Sent
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Sent": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    topic_sizes = topic_sizes.drop(index=0) #ignore the outliers
    topic_sizes.reset_index(inplace = True)
    return topic_sizes
top_n_words = extract_top_n_words_per_topic(tf_idf, count, sent_per_topic, n=10)
topic_sizes = extract_topic_sizes(sent_df); #topic_sizes.head(10)

print(len(topic_sizes))

topics =[]
for i in range(len(topic_sizes)):
    firstList, secondList = list(map(list, zip(*top_n_words[topic_sizes["Topic"][i]])))
    topics.append(firstList)

#print(len(topics))

## print the time required for the modelling in seconds
print("--- %s seconds ---" % (time.time() - start_time))

with open("Topics", "wb") as fp:
  pickle.dump(topics, fp)