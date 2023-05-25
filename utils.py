import numpy as np
import generator

def generate_starting_population(size, task_number, seed):
    population = np.zeros((size, task_number)).astype(int)
    p, w, d = generator.generate_smtwt_instance(task_number, seed)
    
    for i in range(size):
        population[i] = np.arange(task_number)
        np.random.shuffle(population[i])
        
    return population, p, w, d