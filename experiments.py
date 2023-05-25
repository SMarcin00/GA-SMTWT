import generator
import numpy as np
import genetic_algorithm as ga
import random

SEED = 1
TASK_NUMBER = 40
POPULATION_SIZE = 100
MUTATION_PROBABILITY = 0.2
T_SIZE = 50
ITERATIONS = 20

np.random.seed(SEED)
random.seed(SEED)