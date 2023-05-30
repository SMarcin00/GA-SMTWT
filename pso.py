from random import random
from random import seed as set_random_seed
from random import uniform

import pandas as pd
from tqdm import tqdm


class Particle:
    def __init__(self, bounds, n_dimensions, inertia_weight, cognitive_c, social_c):
        self.n_dimensions = n_dimensions
        self.inertia_weight = inertia_weight
        self.cognitive_c = cognitive_c
        self.social_c = social_c

        self.position_i = []
        self.velocity_i = []
        self.pos_best_i = []
        self.err_best_i = -1
        self.err_i = -1

        for i in range(0, self.n_dimensions):
            self.velocity_i.append(uniform(-1, 1))
            self.position_i.append(uniform(bounds[i][0], bounds[i][1]))

    def evaluate(self, cost_function):
        self.err_i = cost_function(self.position_i)

        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i.copy()
            self.err_best_i = self.err_i

    def update_velocity(self, pos_best_g):
        for i in range(0, self.n_dimensions):
            r_l = random()
            r_g = random()

            vel_cognitive = (
                self.cognitive_c * r_l * (self.pos_best_i[i] - self.position_i[i])
            )
            vel_social = self.social_c * r_g * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = (
                self.inertia_weight * self.velocity_i[i] + vel_cognitive + vel_social
            )

    def update_position(self, bounds):
        for i in range(0, self.n_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]


def minimize(
    cost_function,
    dimensions,
    bounds,
    inertia_weight,
    cognitive_c,
    social_c,
    num_particles,
    max_iter,
    verbose=False,
):
    err_best_g = -1
    pos_best_g = []

    swarm = []
    for i in range(0, num_particles):
        swarm.append(
            Particle(bounds, dimensions, inertia_weight, cognitive_c, social_c)
        )

    i = 1
    while i <= max_iter:
        for j in range(0, num_particles):
            swarm[j].evaluate(cost_function)

            if swarm[j].err_i < err_best_g or err_best_g == -1:
                pos_best_g = list(swarm[j].position_i)
                err_best_g = float(swarm[j].err_i)

        for j in range(0, num_particles):
            swarm[j].update_velocity(pos_best_g)
            swarm[j].update_position(bounds)

        if verbose:
            print(f"iter: {i:>4d}, best solution: {err_best_g:10.12f}")

        i += 1

    if verbose:
        print("\nFINAL SOLUTION:")
        print("\tPositions: ", pos_best_g)
        print("\tBest Solution: ", err_best_g)

    particle_positions = [particle.position_i for particle in swarm]
    particle_objective_values = [particle.err_i for particle in swarm]

    return pos_best_g, err_best_g, particle_positions, particle_objective_values


# objective Function
def sphere(X):
    return sum([x**2 for x in X])


def main():
    N_DIMENSIONS = 2

    BOUNDS = [(-100, 100)] * N_DIMENSIONS

    SEEDS = [1, 42, 2137, 119]
    INERTIA_WEIGHTS = [0.2, 0.5, 0.75, 1]
    COGNITIVE_CS = [0.2, 0.5, 0.75, 1]
    SOCIAL_CS = [0.2, 0.5, 0.75, 1]
    NUM_PARTICLES = [15, 30, 60, 100]
    MAX_ITERS = [30, 60, 100, 200]

    pbar = tqdm(
        total=len(SEEDS)
        * len(INERTIA_WEIGHTS)
        * len(COGNITIVE_CS)
        * len(SOCIAL_CS)
        * len(NUM_PARTICLES)
        * len(MAX_ITERS),
        desc="Constant combinations",
    )

    pso_result_rows = []
    for seed in SEEDS:
        set_random_seed(seed)
        for inertia_weight in INERTIA_WEIGHTS:
            for cognitive_c in COGNITIVE_CS:
                for social_c in SOCIAL_CS:
                    for num_particles in NUM_PARTICLES:
                        for max_iter in MAX_ITERS:
                            (
                                pos_best_g,
                                err_best_g,
                                particle_pos,
                                particle_objective_values,
                            ) = minimize(
                                cost_function=sphere,
                                dimensions=N_DIMENSIONS,
                                bounds=BOUNDS,
                                inertia_weight=inertia_weight,
                                cognitive_c=cognitive_c,
                                social_c=social_c,
                                num_particles=num_particles,
                                max_iter=max_iter,
                            )

                            pso_result_rows.append(
                                {
                                    "Seed": seed,
                                    "Dimensions": N_DIMENSIONS,
                                    "NumParticles": num_particles,
                                    "MaxIter": max_iter,
                                    "InertiaWeight": inertia_weight,
                                    "CognitiveC": cognitive_c,
                                    "SocialC": social_c,
                                    "ParticlePositions": particle_pos,
                                    "ParticleObjectiveValues": particle_objective_values,
                                    "BestObjectiveValue": err_best_g,
                                    "BestPosition": pos_best_g,
                                }
                            )
                            pbar.update()

    pso_results_df = pd.DataFrame.from_records(pso_result_rows)
    pso_results_df.to_csv("./data/pso_results.csv", index=False)


if __name__ == "__main__":
    main()
