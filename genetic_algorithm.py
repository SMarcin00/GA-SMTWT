import numpy as np
import random


class Genetic_algoritm:

    def __init__(self, population, p, w, d, t_size, mutation_probability, iterations) -> None:
        self.population = population
        self.p = p
        self.w = w
        self.d = d
        self.scores = None
        self.best_individual_score = float('inf')
        self.best_individual = None
        self.fitness_score()
        self.t_size = t_size
        self.mutation_probability = mutation_probability
        self.iterations = iterations
        self.best_array = []
        self.score_array = []
        self.population_score_array = []
        
        
    def fitness_function(self, individual):
        task_number = len(individual)
        C = np.zeros(task_number).astype(int)
        C[0] = self.p[individual[0]]
        
        for i in range(1, task_number):
            machine = individual[i]
            C[i] = C[i-1] + self.p[machine]
            
        T = np.zeros(task_number).astype(int)
        
        for i in range(task_number):
            T[individual[i]] = max(0, C[i] - self.d[individual[i]])
            
        return np.sum(self.w * T)
    
    
    def fitness_score(self):
        self.scores = np.array([self.fitness_function(x) for x in self.population])
        best_score = np.argmin(self.scores)
        if best_score < self.best_individual_score:
            self.best_individual_score = best_score
            self.best_individual = self.population[best_score]

    
    # def check_individual(self, individual):
    #     new_score = self.fitness_function(individual)
    #     if new_score < self.best_individual_score:
    #         self.best_individual = individual
    #         self.best_individual_score = new_score
    
    
    def tournament_selection(self):
        indexes = random.sample([x for x in range(len(self.population))], self.t_size)
        tournament_population_scores = np.array([self.scores[i] for i in indexes])
        best_individual_index = list(self.scores).index(np.amin(tournament_population_scores))
        return self.population[best_individual_index]
        
    
    def crossover(self, individual1, individual2):       
        size = len(individual1)
        
        index1 = np.random.randint(0, size-2)
        index2 = np.random.randint(index1+1, size-1)
        
        # crossing        
        new_individual1 = np.zeros(size).astype(int)
        new_individual2 = np.zeros(size).astype(int)
        for i in range(size):
            if i in range(index1, index2):
                new_individual1[i] = individual1[i]
                new_individual2[i] = individual2[i]
            else:
                if individual2[i] in individual1[index1:index2]:
                    new_individual1[i] = individual2[list(individual1).index(individual2[i])]
                else:
                    new_individual1[i] = individual2[i]
                if individual1[i] in individual2[index1:index2]:
                    new_individual2[i] = individual1[list(individual2).index(individual1[i])]
                else:
                    new_individual2[i] = individual1[i]
        
        # self.check_individual(new_individual1)
        # self.check_individual(new_individual2)
        return new_individual1, new_individual2
        
    
    def mutation(self, individual):
        if np.random.rand() < self.mutation_probability:
            individual = individual.copy()
            size = len(individual)
            index1 = np.random.randint(0, size-2)
            index2 = np.random.randint(index1+1, size-1)
            
            tmp = individual[index1]
            individual[index1] = individual[index2]
            individual[index2] = tmp
            
            # self.check_individual(individual)
            
        return individual
        
        
    
    def run(self):
        population_size = len(self.population)
        individual_size = len(self.population[0])
        for i in range(self.iterations):
            new_population = np.zeros((population_size, individual_size)).astype(int)
            # choosing parents
            parents = np.zeros((population_size, individual_size)).astype(int)
            for i in range(0, population_size, 2):
                parents[i] = self.tournament_selection()
                parents[i+1] = self.tournament_selection()
                # parent = parents[i]
                # while(parent.all() == parent[i+1].all()):
                #     parents[i+1] = self.tournament_selection()

            # crossing
            for i in range(0, population_size, 2):
                new_population[i], new_population[i+1] = self.crossover(parents[i], parents[i+1])
            self.population = new_population
            
            # remember best individual
            self.fitness_score()
            
            # mutation
            for i in range(population_size):
                self.population[i] = self.mutation(self.population[i])
            
            # remember best individual
            self.fitness_score()
            
            self.best_array.append(self.best_individual)
            self.score_array.append(self.best_individual_score)
            self.population_score_array.append(sum(self.scores))
            
            
       
        