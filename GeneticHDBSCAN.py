import random
from copy import deepcopy

import hdbscan
from sklearn.metrics import silhouette_score


def objective_fun(genes, X):
    
    eps = genes[0].gene
    min_samples = genes[1].gene
    

    labels = hdbscan.HDBSCAN(min_cluster_size=5,
                          min_samples=min_samples,
                          cluster_selection_epsilon=eps,
                          metric='euclidean',
                          cluster_selection_method='eom').fit(X)
    
    try:
        
        return silhouette_score(X, labels)
    
    except:
        
        return 1e-10
    


class Gene:

    X = None

    def __init__(self, gene, name):
        
        self.gene = gene
        self.name = name
        
        
    def __add__(self, gene) -> float:
        
        return objective_fun([self, gene], Gene.X)
    
        
    def __eq__(self, gene) -> bool:
        
        return self.gene == gene.gene
        
        
    def __repr__(self) -> str:
        
        return f"{self.name} = {self.gene}"
        
    


class Cromosom:
    
    def __init__(self, cromo:list[Gene]):
        
        self.cromo = cromo
        self.cost = float('inf')
        self.cromo_len = len(self.cromo)
        
        self.get_cost()
        
        
    def get_cost(self):
        
        for i in range(self.cromo_len - 1):
            self.cost += self.cromo[i] + self.cromo[i + 1]
            
    
    def mutet_cromosom(self, eps, nim_sample):
        
        gen_num = random.randint(0, self.cromo_len - 1)
        
        
        self.cromo[gen_num] = Gene(int(random.uniform(nim_sample[0], nim_sample[1])), 'min_samples') \
                                    if gen_num \
                                    else Gene(random.uniform(eps[0], eps[1]), 'eps')
        
        self.get_cost()
        
    
    
    def __add__(self, cromosom):
        
        child_cromo = deepcopy(self.cromo) 
        
        gen_num = random.randint(0, self.cromo_len - 1)
        
        child_cromo[gen_num] = deepcopy(cromosom[gen_num])
        
        return Cromosom(child_cromo)
    
    
    def __len__(self):
        return self.cromo_len
    
    def __getitem__(self,index):
         return self.cromo[index]
        
    
    def __gt__(self, cromosom):
        
        return self.cost > cromosom.cost
    
    
    def __lt__(self, cromosom):
        
        return self.cost < cromosom.cost
    
    
    def __eq__(self, cromosom):
        
        return self.cost == cromosom.cost
    
    
    def __repr__(self):
        out = ' , '.join([str(gene) for gene in self.cromo])
        return out
    


class Population:
    
    def __init__(self, genes_keys, population_size=200, elition_size=30, crossOver_propa=0.5, mutation_propa=0.1):
        
        self.genes_keys = genes_keys
        
        self.population_size = population_size
        self.elition_size = elition_size
        self.crossOver_propa = crossOver_propa
        self.mutation_propa = mutation_propa
        self.old_population = []
        self.new_population = []

        random.seed(32)

        self.make_population()
        
        
    def make_population(self, print_state=False):
        
        for i in range(self.population_size):
            
            eps = random.uniform(self.genes_keys[0][0], self.genes_keys[0][1])
            min_sample = int(random.uniform(self.genes_keys[1][0], self.genes_keys[1][1]))
            
            self.old_population.append(Cromosom([Gene(eps, 'eps'), Gene(min_sample, 'min_samples')]))
            
        if print_state:
            return self.old_population
        
        
    def elite_populatio(self):
        
        elite = list(sorted(self.old_population, reverse=True))[:self.elition_size]
        
        self.new_population += elite
        
        
        
    def cross_over(self):
        
        
        for i in range(0, self.population_size, 2):
            
            cromo1_id = random.randrange(self.population_size)
            cromo2_id = random.randrange(self.population_size)
            
            
            cromo1 = deepcopy(self.old_population[cromo1_id])
            cromo2 = deepcopy(self.old_population[cromo2_id])
            
            
            cross_propa = random.random()
            
            
            
            if cross_propa < self.crossOver_propa:
                
                cromo1 = self.old_population[cromo1_id] + self.old_population[cromo2_id]
                
                cromo2 = self.old_population[cromo2_id] + self.old_population[cromo1_id]
                
            
            
            mutet_propa = random.random()
            
            if mutet_propa > self.mutation_propa:
                
                cromo1.mutet_cromosom(self.genes_keys[0], self.genes_keys[1])
                cromo2.mutet_cromosom(self.genes_keys[0], self.genes_keys[1])
                
                
            self.new_population.append(cromo1)
            self.new_population.append(cromo2)
            
            
    def nex_iter(self):
        self.old_population = deepcopy(self.new_population)
        self.new_population = []
        