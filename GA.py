import pygad
import numpy as np
from time import time
from numpy.random import rand
from torch import tensor, Tensor
from torch.distributions import MultivariateNormal
from pandas import DataFrame
from src2.toolbox.CustomLogger import CustomLogger

SEEDS = [42, 18, 2306406, 98, 12471, 41274, 5328, 1247, 125732,31253]

record = []
for N_dims in range(5,25,5) : 
    for num_population_ratio in [0.2,0.5,1,2,4] : 
        num_population = max(int(num_population_ratio * N_dims), 5)
        for num_parents_mating_ratio in [0.1,0.2,0.5] : 
            num_parents_mating = max(int(num_parents_mating_ratio * num_population), 2)
            for elitism_ratio in [0,0.5,1] : 
                n_elitism = int(num_parents_mating * elitism_ratio)
                for seed in SEEDS:
                    passing = [False, False, False]
                    try : 
                        mu = tensor([10 * rand() for _ in range(N_dims)])
                        cov = tensor([[(i==j) * (rand() + 15) + 0.5 for i in range(N_dims)] for j in range(N_dims)])
                        f = MultivariateNormal(mu, cov)

                        def fitness_func(ga_instance, solution : np.ndarray, solution_idx : int):
                            x = [float(el) for el in solution]
                            out : Tensor = f.log_prob(tensor(x))
                            return float(out)
                        
                        passing[0] = True

                        GA_parameters = {
                            #Must Specify
                            'fitness_func' : fitness_func,
                            'num_generations' : 5000,
                            
                            'sol_per_pop' : num_population,
                            'num_parents_mating' : num_parents_mating,
                            'keep_elitism' : n_elitism,
                            
                            'num_genes' : N_dims,
                            "gene_space" : [
                                {'low' : -20, 'high' : 20}
                                for _ in range(N_dims)
                            ],
                            # "gene_type": [float for _ in range(N_dims)],
                            "stop_criteria" : "saturate_20",
                            # Default
                            'mutation_type' : "random",
                            'parent_selection_type' : "sss",
                            'crossover_type' : "single_point",
                            'mutation_percent_genes' : max(int(10 / N_dims) * 10, 10),
                            # Other
                            'save_solutions' : False,
                            'random_seed' : seed
                        }

                        ga_instance = pygad.GA(**GA_parameters)
                        passing[1] = True

                        t1 = time()
                        ga_instance.run()
                        t2 = time()
                        passing[2] = True

                        solution, value, _ = ga_instance.best_solution()
                        norme = np.linalg.norm(solution - mu.numpy())
                        generation = max(ga_instance.generations_completed - 20, 1)
                        time_completion = t2 - t1
                    
                    except : 
                        norme = np.nan
                        generation = np.nan
                        time_completion = np.nan
                    
                    finally : 
                        record.append({
                            'err' : norme,
                            'generation' : generation,
                            'time' : time_completion,

                            'N_dims' : N_dims,
                            
                            'n_population_ratio' : num_population_ratio,
                            'n_population' : num_population,

                            'num_parents_mating_ratio' : num_parents_mating_ratio,
                            'num_parents_mating' : num_parents_mating,

                            'elitism_ratio' : elitism_ratio,
                            'n_elitism' : n_elitism,
                            
                            'seed' : seed,
                            'passing' : passing
                        })
                break
            break
        break
    break

DataFrame(record).to_csv("GA_exploration.csv", index=False)
CustomLogger().notify_when_done()