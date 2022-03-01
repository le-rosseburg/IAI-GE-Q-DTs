"""
Implementation of the grammatical evolution

"""
import numpy as np
from deap import base, creator, tools
from deap_algorithms import eaSimple


def grammatical_evolution(
    fitness_function,
    n_individuals=200,
    n_generations=100,
    cxpb=0,
    mutpb=1,
    genotype_len=1024,
    selection={"function": "tools.selTournament", "tournsize": 2},
    mutation={"function": "tools.mutUniformInt", "low": 0, "up": 40000, "indpb": 0.1},
    crossover={"function": "tools.cxOnePoint"},
    seed=42,
    logfile=None,
):
    """
    Implementation of the grammatical evolution using the deap library.
    Creates and defines the toolbox and stats and applies the evolutionary algorithm 'eaSimple'.

    :param (function) fitness_function: The fitness function for the evolutionary algorithm
    :param (int) n_individuals: Number of individuals inside a population
    :param (int) n_generations: Number of generations used for the evolutionary algorithm
    :param (float) cxpb: The crossover probability
    :param (float) mutpb: The mutation probability
    :param (int) genotype_len: The fixed length of a genotyp
    :param (dict) selection: A Dictionary with the function and its parameters used for the selection inside the evolutionary algorithm
    :param (dict) mutation: A Dictionary with the function and its parameters used for the mutation inside the evolutionary algorithm
    :param (dict) crossover: A Dictionary with the function and its parameters used for the mating inside the evolutionary algorithm
    :param (int) seed: The used seed
    :param (string) logfile: The path to the logfile

    Returns:
        (list<deap.creator.Individual>) pop: The final population
        (deap.tools.support.HallOfFame) hof: The best individual (with the best fitness value) that ever lived in the population
        (deap.tools.support.Logbook) log: The logbook returned from the evolutionary algorithm 'eaSimple'
        (dict<string, float>) best_leaves:
    """
    np.random.seed(seed)

    # define deap types and tools
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # codons a.k.a. genes as integers
    toolbox.register("attribute_generator", np.random.randint, 0, 40000)
    # list of codons a.k.a. genes of fixed length, bsp. [4,2,...,6]
    toolbox.register(
        "individual_generator",
        tools.initRepeat,
        creator.Individual,
        toolbox.attribute_generator,
        genotype_len,
    )
    # list of individuals
    toolbox.register(
        "population_generator", tools.initRepeat, list, toolbox.individual_generator
    )
    toolbox.register("evaluate", fitness_function)
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

    # generate population and hall of fame
    pop = toolbox.population_generator(n_individuals)
    hof = tools.HallOfFame(1)

    # create and define statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # apply evolutionary algorithm
    pop, log, best_leaves = eaSimple(
        population=pop,
        toolbox=toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=n_generations,
        stats=stats,
        halloffame=hof,
        verbose=True,
        logfile=logfile,
    )

    return pop, log, hof, best_leaves
