import hdbscan
import pickle
from operator import itemgetter
from sklearn.metrics import silhouette_score
import random
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

def GA_AddNC(data):

    # Read Populstion
    with open("Generation60", "rb") as fp:
        Pop = pickle.load(fp)

    # Sort According to Fitness
    Pop = sorted(Pop, key=itemgetter(5), reverse=True)

    Pop = objective_fun(Pop, data)
    print(Pop)

    with open("Generation60", "wb") as fp:
        pickle.dump(Pop, fp)




def GI_Population(min_size, min_samples, eps, data):

    Pop_Size = 50
    Pop = GA_Population(min_size, min_samples, eps, data, Pop_Size)
    with open("Population2", "wb") as fp:
        pickle.dump(Pop, fp)
    print(Pop)

def GA_HDBSCAN(min_size, min_samples, eps, data):

    # Read Populstion
    with open("Generation60", "rb") as fp:
        Pop = pickle.load(fp)
    print(len(Pop))

    # Sort According to Fitness
    Pop = sorted(Pop, key=itemgetter(5), reverse=True)
    print(Pop)
    
    '''
    # Start Generations
    for i in range(10):
        Gen = Pop[:20]

        # Crossover Step
        NewGen = []
        for j in range(0,20,2):

            Sol1 = Gen[j]
            Sol2 = Gen[j+1]
            #print(Sol1, "  ",Sol2)
            
            NewSol = []
            NewSol.append(round((Sol1[0]+Sol2[0])/2))
            NewSol.append(round((Sol1[1]+Sol2[1])/2))
            NewSol.append((Sol1[2]+Sol2[2])/2)
            #print(NewSol)

            # Mutation Step
            k = random.randint(30,60)
            if k in range(51, 53):
                if k == 51:
                    NewSol[0] = random.randint(100, 200)
                
                elif k == 52:
                    NewSol[1] = random.randint(20, 100)
                
                else:
                    NewSol[2] = round(random.uniform(eps[0],eps[1]), 1)
            NewGen.append(NewSol)
        #print(NewGen)
        NewGen = objective_fun(NewGen, data)
        #print(NewGen)


        # Replacement Step
        Pop = Pop[:len(Pop)-10]
        Pop = Pop + NewGen
        #print("length of new Pop: ", len(Pop))

    #print("the length of Gen 5 is: ", len(Pop))
    with open("Generation60", "wb") as fp:
        pickle.dump(Pop, fp)
    '''
    

def GA_Population(min_size, min_samples, eps, data, Pop_Size):
    Pop = []
    for i in range(Pop_Size):
        sol = []
        mSize = random.randint(min_size[0], min_size[1])
        sol.append(mSize)
        mSample = random.randint(min_samples[0], min_samples[1])
        sol.append(mSample)
        e = round(random.uniform(eps[0],eps[1]), 1)
        sol.append(e)
        Pop.append(sol)
    
    Pop = objective_fun(Pop, data)

    return Pop


def objective_fun(Pop, data):

    for i in range(len(Pop)):

        sol = Pop[i]
        mSize = sol[0]
        mSample = sol[1]
        eps = sol[2]
        clusters = hdbscan.HDBSCAN(min_cluster_size=mSize,
                            min_samples=mSample,
                            cluster_selection_epsilon=eps,
                            metric='euclidean',
                            cluster_selection_method='eom').fit(data)
        topics = Get_Topics(clusters)
        TC, TDP, TQ, CN = EvaluateSol(topics)
        #Pop[i].append(TC)
        #Pop[i].append(TDP)
        #Pop[i].append(TQ)
        Pop[i].append(CN)

        #Pop.append(silhouette_score(data, clusters.labels_))
    return Pop
    

if __name__ == '__main__':
    min_size = [100, 300]
    min_samples = [20, 100]
    eps = [0.2, 1.0]

    #GI_Population(min_size, min_samples, eps, embeddings)

    #GA_HDBSCAN(min_size, min_samples, eps, embeddings)
    #x, y = GA_HDBSCAN(min_samples, eps, embeddings)

    GA_AddNC(embeddings)

    #print(x," - ", y)
