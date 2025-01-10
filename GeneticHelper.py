
from sklearn.decomposition import PCA
import hdbscan
import pickle
from sklearn.metrics import silhouette_score

from GeneticHDBSCAN import Population, Gene

def get_parameters(eps_range, min_samples_range, X):
    Gene.X = X
    data = [eps_range, min_samples_range]
    population = Population(data)

    for _ in range(50):
        population.elite_populatio()
        population.cross_over()
        population.nex_iter()

    return max(population.old_population)
