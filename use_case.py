import generator
import numpy as np
import genetic_algorithm as ga
import random

def generate_starting_population(size, task_number, seed):
    population = np.zeros((size, task_number)).astype(int)
    p, w, d = generator.generate_smtwt_instance(task_number, seed)
    
    for i in range(size):
        population[i] = np.arange(task_number)
        np.random.shuffle(population[i])
        
    return population, p, w, d


SEED = 1
TASK_NUMBER = 40
POPULATION_SIZE = 100
MUTATION_PROBABILITY = 0.2
T_SIZE = 50
ITERATIONS = 20

np.random.seed(SEED)
random.seed(SEED)

population, p, w, d = generate_starting_population(
    size=POPULATION_SIZE, 
    task_number=TASK_NUMBER, 
    seed=SEED)


gen_alg = ga.Genetic_algoritm(population, p, w, d, T_SIZE, MUTATION_PROBABILITY, iterations=ITERATIONS)
gen_alg.run()
print(gen_alg.best_individual)
print(gen_alg.score_array)
print(gen_alg.population_score_array)
print('end')


