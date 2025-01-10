import hdbscan
import pickle
from operator import itemgetter
from sklearn.metrics import silhouette_score
import random
from sklearn.cluster import MiniBatchKMeans
from GeneticHDBSCAN import Population, Gene
from sklearn.metrics import silhouette_score

from GA_Topics import Get_Topics
from GA_Evaluations import EvaluateSol

import GeneticHelper as hf

###########################################################
## Load the Umap Embeddings
###########################################################

with open("embeddings", "rb") as fp:
  embeddings = pickle.load(fp)

###########################################################

def GI_Population(n_clusters, batch_size, n_init, data):

    Pop_Size = 50
    Pop = GA_Population(n_clusters, batch_size, n_init, data, Pop_Size)
    with open("MBPop0", "wb") as fp:
        pickle.dump(Pop, fp)
    print(Pop)

def GA_MBKM(n_clusters, batch_size, n_init, data):

    # Read Populstion
    with open("MBPop50", "rb") as fp:
        Pop = pickle.load(fp)
    print(len(Pop))

    # Sort According to Fitness
    Pop = sorted(Pop, key=itemgetter(5), reverse=True)
    print(Pop)
    
    '''
    # Start Generations
    for i in range(20):
        Gen = Pop[:20]

        # Crossover Step
        NewGen = []
        for j in range(0,20,2):

            Sol1 = Gen[j]
            Sol2 = Gen[j+1]
            #print(Sol1, "  ",Sol2)
            
            NewSol = []
            NewSol.append(round((Sol1[0]+Sol2[0])/2))
            MBS = (Sol1[1]+Sol2[1])/2
            NBS = min(batch_size, key=lambda x:abs(x-MBS))
            NewSol.append(NBS)
            NewSol.append(round((Sol1[2]+Sol2[2])/2))
            #print(NewSol)

            # Mutation Step
            k = random.randint(1,100)
            if k in range(51, 53):
                if k == 51:
                    NewSol[0] = random.randint(10, 20)
                
                elif k == 52:
                    NewSol[1] = random.choice(batch_size)
                
                else:
                    NewSol[2] = random.randint(1, 10)
            NewGen.append(NewSol)
        #print(NewGen)
        NewGen = objective_fun(NewGen, data)
        #print(NewGen)


        # Replacement Step
        Pop = Pop[:len(Pop)-10]
        Pop = Pop + NewGen
        #print("length of new Pop: ", len(Pop))

    #print("the length of Gen 5 is: ", len(Pop))
    with open("MBPop50", "wb") as fp:
        pickle.dump(Pop, fp)
    '''
    

def GA_Population(n_clusters, batch_size, n_init, data, Pop_Size):
    Pop = []
    for i in range(Pop_Size):
        sol = []
        nC = random.randint(n_clusters[0], n_clusters[1])
        sol.append(nC)
        bS = random.choice(batch_size)
        sol.append(bS)
        nI = random.randint(n_init[0], n_init[1])
        sol.append(nI)
        Pop.append(sol)
    
    Pop = objective_fun(Pop, data)

    return Pop


def objective_fun(Pop, data):

    for i in range(len(Pop)):

        sol = Pop[i]
        nC = sol[0]
        bS = sol[1]
        nI = sol[2]
        clusters = MiniBatchKMeans(n_clusters=nC,
                          init='k-means++',
                          random_state=0,
                          batch_size=bS,
                          max_iter=200,
                          n_init=nI).partial_fit(data)
        topics = Get_Topics(clusters)
        TC, TDP, TQ, CN = EvaluateSol(topics)
        Pop[i].append(TC)
        Pop[i].append(TDP)
        Pop[i].append(TQ)
        #Pop[i].append(CN)

        #Pop.append(silhouette_score(data, clusters.labels_))
    return Pop
    

if __name__ == '__main__':
    n_clusters = [10, 20]
    batch_size = [256, 512, 1024]
    n_init = [1, 10]

    #GI_Population(n_clusters, batch_size, n_init, embeddings)

    GA_MBKM(n_clusters, batch_size, n_init, embeddings)
    #x, y = GA_MBKM(min_samples, eps, embeddings)

    #print(x," - ", y)
