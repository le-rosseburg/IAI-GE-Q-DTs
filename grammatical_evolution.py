"""
Implementation of the grammatical evolution

"""
import numpy as np
from deap import base, creator, tools, algorithms
from deap_algorithms import eaSimple


def grammatical_evolution(
    fitness_function,
    n_individuals=200,
    n_generations=100,
    jobs=1,
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
    :param (int) jobs: Number oj jobs used when multiprocessing is activated
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


# die funktionen werden überhaupt nicht benutzt.
# Es wird auf die default funktionen zugegriffen die beim parser angegeben sind
# das sind:
# mutation: tools.mutUniformInt#low-0#up-40000#indpb-0.1
# crossover: tools.cxOnePoint
# selection: tools.selTournament#tournsize-2

"""# mutation
def mutate(ind, attribute):
    return tools.mutUniformInt(ind, 0, 40000, 0.1)

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


# crossover, only used inside the oblique tests
def mate(ind1):
    offspring = tools.cxOnePoint(ind1, ind2)

    # makes no sense
    if np.random.uniform() < 0.5:
        new_offspring = []
        for idx, ind in enumerate([ind1, ind2]):
            _, used = GETranslator(1, [object], [0]).genotype_to_str(ind)
            if used > len(ind):
                used = len(ind)
            # generates a new genotype with the length min(length(ind1), length(ind2))
            new_offspring.append(individual(offspring[idx][:used]))
        offspring = (new_offspring[0], new_offspring[1])
    return offspring"""
