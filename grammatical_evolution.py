#!/usr/bin/python3
"""
Implementation of the grammatical evolution

"""
import numpy as np
from deap import base, creator, tools, algorithms
from deap_algorithms import eaSimple


def grammatical_evolution(
    fitness_function,
    inputs,
    leaf,
    n_individuals,
    n_generations,
    jobs,
    cxpb,
    mutpb,
    init_len=100,
    selection={"function": "tools.selBest"},
    mutation={"function": "mutate", "attribute": None},
    crossover={"function": "mate", "individual": None},
    seed=42,
    logfile=None,
):
    np.random.seed(seed)
    max_v = 40000

    # define deap types and tools
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create(
        "Individual", list, fitness=creator.FitnessMax
    )

    # list of codons, bsp. [4,2,...,6]
    toolbox = base.Toolbox()
    toolbox.register(
        "attribute_generator", np.random.randint, 0, max_v
    )  # genes as integers
    toolbox.register(
        "individual_generator",
        tools.initRepeat,
        creator.Individual,
        toolbox.attribute_generator,
        init_len,
    )  # list of codons/genes of fixed length
    toolbox.register(
        "population_generator", tools.initRepeat, list, toolbox.individual_generator
    )  # list of individuals
    toolbox.register("evaluate", fitness_function)

    # assign created deap types/tools to functions
    for d in [mutation, crossover]:
        if "attribute" in d:
            d["attribute"] = toolbox.attr_bool
        if "individual" in d:
            d["individual"] = creator.Individual

    toolbox.register(
        "mutate",
        eval(mutation["function"]),
        **{k: v for k, v in mutation.items() if k != "function"}
    )
    toolbox.register(
        "mate",
        eval(crossover["function"]),
        **{k: v for k, v in crossover.items() if k != "function"}
    )
    toolbox.register(
        "select",
        eval(selection["function"]),
        **{k: v for k, v in selection.items() if k != "function"}
    )

    # generate population
    pop = toolbox.population_generator(n_individuals)
    hof = tools.HallOfFame(1)

    # create statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # apply evolutionary algorithm
    pop, log, best_leaves = eaSimple(
        pop,
        toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=n_generations,
        stats=stats,
        halloffame=hof,
        verbose=True,
        logfile=logfile,
    )

    return pop, log, hof, best_leaves


"""commented out definitions are my interpretations of the functions used/described in paper"""

# mutation
def mutate(ind, attribute):
    rand = np.random.randint(0, len(ind) - 1)
    assert rand >= 0

    if np.random.uniform() < 0.5:
        # randomly mutate one gene
        ind[rand] = attribute_generator()
    else:
        # makes no sense as genotypes should be of fixed lengths according to paper
        # duplicates a random amount of genes
        ind.extend(np.random.choice(ind, size=rand))
    return (ind,)


"""def mutate(ind, attribute):
  rand = np.random.randint(0, len(ind) - 1)
  assert rand >= 0

  ind[rand] = attribute()
  return ind, """


# crossover
def mate(ind1, ind2, individual):
    offspring = tools.cxOnePoint(ind1, ind2)

    # makes no sense
    if np.random.uniform() < 0.5:
        new_offspring = []
        for idx, ind in enumerate([ind1, ind2]):
            _, used = GETranslator(1, [object], [0]).genotype_to_str(ind)
            if used > len(ind):
                used = len(ind)
            # generates a new genotype with the length min(length(ind1), length(ind2))
            new_offspring.append(individual_generator(offspring[idx][:used]))
        offspring = (new_offspring[0], new_offspring[1])
    return offspring


"""def mate(ind1, ind2, individual):
  offspring = tools.cxOnePoint(ind1, ind2)
  return offspring"""
