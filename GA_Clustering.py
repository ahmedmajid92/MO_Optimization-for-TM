import hdbscan
import pickle
from sklearn.metrics import silhouette_score

from GeneticHDBSCAN import Population, Gene
from sklearn.metrics import silhouette_score

import GeneticHelper as hf

###########################################################
## Load the Umap Embeddings
###########################################################

with open("embeddings", "rb") as fp:
  X = pickle.load(fp)

###########################################################


def submit_values(eps1, eps2, min_samples1, min_samples2):
    print(eps1, eps2, min_samples1, min_samples2)

    if eps1 and eps2 and min_samples1 and  min_samples2:
        # print('here1')
        eps, min_sample = hf.get_parameters([eps1, eps2], [min_samples1, min_samples2], X)
        # print('here2', eps, min_sample)
        labels = hdbscan.HDBSCAN(min_cluster_size=5,
                                 min_samples=min_sample,
                                 cluster_selection_epsilon=eps,
                                 metric='euclidean',
                                 cluster_selection_method='eom').fit(X)
        silhouette = 0

        try:
            print('here3')
            silhouette = silhouette_score(X, labels)
        except:
            print('wrong')

if __name__ == '__main__':
    # run app
    submit_values(0.2,1,1,50)