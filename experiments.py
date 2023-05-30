import generator
import numpy as np
import genetic_algorithm as ga
import random
from tqdm import tqdm
from utils import generate_starting_population
import pandas as pd

SEED = 1
TASK_NUMBER = 30
POPULATION_SIZE = 30
MUTATION_PROBABILITY = 0.1
T_SIZE = 3
ITERATIONS = 20

np.random.seed(SEED)
random.seed(SEED)

seed_values = [1, 42, 321, 2137, 321, 1435]
population_sizes = [10, 20, 30, 40, 50, 60]
mutation_probabilities = [x/100 for x in range(5, 95, 5)]
t_sizes = [x for x in range(2,30,2)]
iterations = [10, 30, 50, 70, 90]

print(" Genetic Algorithm".center(80, "="))
def experiment_population():
    sa_pbar = tqdm(total=(len(seed_values)*len(population_sizes)))
    print(" Population Sizes".center(80, "="))
    ga_population = []
    for seed in seed_values:
        np.random.seed(seed)
        random.seed(seed)
        for size in population_sizes:
            pop, p, w, d = generate_starting_population(
                size=size, 
                task_number=TASK_NUMBER, 
                seed=seed
            )
            
            gen_alg = ga.Genetic_algoritm(
                population=pop,
                p=p,
                w=w,
                d=d,
                t_size=T_SIZE,
                mutation_probability=MUTATION_PROBABILITY,
                iterations=ITERATIONS,
            )
            gen_alg.run()
            
            ga_population.append(
                            {
                                "Seed": seed,
                                "PopulationSize": size,
                                "TaskNumber": TASK_NUMBER,
                                "tSize": T_SIZE,
                                "MutationProbability": MUTATION_PROBABILITY,
                                "Iterations": ITERATIONS,
                                "BestIndividual": gen_alg.best_individual,
                                "BestIndividualScore": gen_alg.best_individual_score,
                                # "BestScoresArray": gen_alg.score_array,
                                # "PopulationScores": gen_alg.population_score_array
                            }
                        )
            
            sa_pbar.update()
    sa_pbar.close()
        
    ga_population_df = pd.DataFrame.from_dict(ga_population)
    ga_population_out = "data/ga_population.csv"
    ga_population_df.to_csv(ga_population_out)
    print(f"Results of g_a saved to {ga_population_out}\n\n")


def experiment_tsize():
    sa_pbar = tqdm(total=(len(seed_values)*len(t_sizes)))
    print(" Tsizes Sizes".center(80, "="))
    ga_population = []
    for seed in seed_values:
        np.random.seed(seed)
        random.seed(seed)
        for size in t_sizes:
            pop, p, w, d = generate_starting_population(
                size=POPULATION_SIZE, 
                task_number=TASK_NUMBER, 
                seed=seed
            )
            
            gen_alg = ga.Genetic_algoritm(
                population=pop,
                p=p,
                w=w,
                d=d,
                t_size=size,
                mutation_probability=MUTATION_PROBABILITY,
                iterations=ITERATIONS,
            )
            gen_alg.run()
            
            ga_population.append(
                            {
                                "Seed": seed,
                                "PopulationSize": POPULATION_SIZE,
                                "TaskNumber": TASK_NUMBER,
                                "tSize": size,
                                "MutationProbability": MUTATION_PROBABILITY,
                                "Iterations": ITERATIONS,
                                "BestIndividual": gen_alg.best_individual,
                                "BestIndividualScore": gen_alg.best_individual_score,
                                # "BestScoresArray": gen_alg.score_array,
                                # "PopulationScores": gen_alg.population_score_array
                            }
                        )
            
            sa_pbar.update()
    sa_pbar.close()
        
    df = pd.DataFrame.from_dict(ga_population)
    ga_out = "data/ga_tsize.csv"
    df.to_csv(ga_out)
    print(f"Results of g_a saved to {ga_out}\n\n")


def experiment_mutation_probability():
    sa_pbar = tqdm(total=(len(seed_values)*len(mutation_probabilities)))
    print(" Tsizes Sizes".center(80, "="))
    ga_population = []
    for seed in seed_values:
        np.random.seed(seed)
        random.seed(seed)
        for prob in mutation_probabilities:
            pop, p, w, d = generate_starting_population(
                size=POPULATION_SIZE, 
                task_number=TASK_NUMBER, 
                seed=seed
            )
            
            gen_alg = ga.Genetic_algoritm(
                population=pop,
                p=p,
                w=w,
                d=d,
                t_size=T_SIZE,
                mutation_probability=prob,
                iterations=ITERATIONS,
            )
            gen_alg.run()
            
            ga_population.append(
                            {
                                "Seed": seed,
                                "PopulationSize": POPULATION_SIZE,
                                "TaskNumber": TASK_NUMBER,
                                "tSize": T_SIZE,
                                "MutationProbability": prob,
                                "Iterations": ITERATIONS,
                                "BestIndividual": gen_alg.best_individual,
                                "BestIndividualScore": gen_alg.best_individual_score,
                                # "BestScoresArray": gen_alg.score_array,
                                # "PopulationScores": gen_alg.population_score_array
                            }
                        )
            
            sa_pbar.update()
    sa_pbar.close()
        
    df = pd.DataFrame.from_dict(ga_population)
    ga_out = "data/ga_mut_prob.csv"
    df.to_csv(ga_out)
    print(f"Results of g_a saved to {ga_out}\n\n")
    
def experiment_iteration():
    sa_pbar = tqdm(total=(len(seed_values)*len(iterations)))
    print(" Tsizes Sizes".center(80, "="))
    ga_population = []
    for seed in seed_values:
        np.random.seed(seed)
        random.seed(seed)
        for iter in iterations:
            pop, p, w, d = generate_starting_population(
                size=POPULATION_SIZE, 
                task_number=TASK_NUMBER, 
                seed=seed
            )
            
            gen_alg = ga.Genetic_algoritm(
                population=pop,
                p=p,
                w=w,
                d=d,
                t_size=T_SIZE,
                mutation_probability=MUTATION_PROBABILITY,
                iterations=iter,
            )
            gen_alg.run()
            
            ga_population.append(
                            {
                                "Seed": seed,
                                "PopulationSize": POPULATION_SIZE,
                                "TaskNumber": TASK_NUMBER,
                                "tSize": T_SIZE,
                                "MutationProbability": MUTATION_PROBABILITY,
                                "Iterations": iter,
                                "BestIndividual": gen_alg.best_individual,
                                "BestIndividualScore": gen_alg.best_individual_score,
                                # "BestScoresArray": gen_alg.score_array,
                                # "PopulationScores": gen_alg.population_score_array
                            }
                        )
            
            sa_pbar.update()
    sa_pbar.close()
        
    df = pd.DataFrame.from_dict(ga_population)
    ga_out = "data/ga_iter.csv"
    df.to_csv(ga_out)
    print(f"Results of g_a saved to {ga_out}\n\n")
    

experiment_mutation_probability()