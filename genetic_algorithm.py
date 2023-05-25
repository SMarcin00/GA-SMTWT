import numpy as np

class Genetic_algoritm:

    def __init__(self, population, p, w, d) -> None:
        self.population = population
        self.p = p
        self.w = w
        self.d = d
        
    def fitness_function(self, individual):
        individual.astype(int)
        task_number = len(individual)
        C = np.zeros(task_number)
        C[0] = self.p[individual[0]]
        
        for i in range(1, task_number):
            machine = individual[i]
            C[i] = C[i-1] + self.p[machine]
            
        T = np.zeros(task_number)
        
        for i in range(task_number):
            T[individual[i]] = max(0, C[i] - self.d[individual[i]])
        
        return np.sum(self.w * T)
        