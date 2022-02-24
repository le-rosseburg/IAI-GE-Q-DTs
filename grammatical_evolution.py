"""
Implementation of the grammatical evolution

"""
import numpy as np
from deap import base, creator, tools, algorithms
from deap.algorithms import eaSimple, varAnd


def grammatical_evolution(fitness_function, n_individuals, n_generations, cxpb, mutpb, init_len=100, 
                          selection={'function': "tools.selBest"}, mutation={'function': "mutate", 'attribute': None}, 
                          crossover={'function': "mate", 'individual': None}, seed=42, logfile=None):
  np.random.seed(seed)
  max_v = 40000

  creator.create("FitnessMax", base.Fitness, weights=(1.0,))
  creator.create("Individual", list, fitness=creator.FitnessMax)
  toolbox = base.Toolbox()
  toolbox.register("attribute_generator", np.random.randint, 0, max_v)
  toolbox.register("individual_generator", tools.initRepeat, creator.Individual, toolbox.attribute_generator, init_len)
  toolbox.register("population_generator", tools.initRepeat, list, toolbox.individual)
  toolbox.register("evaluate", fitness_function)
  
  #???
  """for d in [mutation, crossover]:
    if "attribute" in d:
        d['attribute'] = toolbox.attr_bool
    if "individual" in d:
        d['individual'] = creator.Individual"""

  #???
  toolbox.register("mutate", eval(mutate['function']), **{k: v for k, v in mutate.items() if k != "function"})
  toolbox.register("mate", eval(crossover['function']), **{k: v for k, v in crossover.items() if k != "function"})  
  toolbox.register("select", eval(selection['function']), **{k: v for k, v in selection.items() if k != "function"})

  pop = toolbox.population_generator(n_individuals)
  hof = tools.HallOfFame(1)

  stats = tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("avg", np.mean)
  stats.register("std", np.std)
  stats.register("min", np.min)
  stats.register("max", np.max)

  pop, log, best_leaves = eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_generations, 
                                   stats=stats, halloffame=hof, verbose=True, logfile=logfile)

  return pop, los, best_leaves

# mutation
def mutate(individual, attribute, prob, max):
  rand = np.random.randint(0, len(individual) - 1)
  assert rand >= 0

  if np.random.uniform() < 0.5:
    individual[rand] = attribute()
  else:
    individual.extend(np.random.choice(individual, size=rand))
  return individual, 

# crossover
def mate():
  return

# selection
def select():
  return

