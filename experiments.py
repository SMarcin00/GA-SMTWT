import generator
import numpy as np
import genetic_algorithm as ga

def generate_starting_population(size, task_number, seed):
    population = np.zeros((size, task_number)).astype(int)
    p, w, d = generator.generate_smtwt_instance(task_number, seed)
    
    for i in range(size):
        population[i] = np.arange(task_number)
        np.random.shuffle(population[i])
        print(population)

    return population, p, w, d


SEED = 1234
TASK_NUMBER = 4
POPULATION_SIZE = 3

population, p, w, d = generate_starting_population(size=POPULATION_SIZE, task_number=TASK_NUMBER, seed=SEED)

gen_alg = ga.Genetic_algoritm(population, p, w, d)
fitness = gen_alg.fitness_function(population[0])

