import os
import gym
import string
import datetime
import argparse
import numpy as np
from time import time, sleep
from numpy import random
from decision_tree import DecisionTree, Leaf
from grammatical_evolution import grammatical_evolution
from ge_translator import GETranslator
from joblib import parallel_backend


def string_to_dict(string):
    """
    This function splits a string into a dict of integers or floats.
    The string must be in the format: key0-value0#key1-value1#...#keyn-valuen
    :param string: The string that gets converted

    Results:
        result: A dict of the input string
    """
    result = {}
    items = string.split("#")

    for i in items:
        key, value = i.split("-")
        try:
            result[key] = int(value)
        except:
            try:
                result[key] = float(value)
            except:
                result[key] = value

    return result


parser = argparse.ArgumentParser()
parser.add_argument(
    "--jobs", default=1, type=int, help="The number of jobs to use for the evolution"
)
parser.add_argument("--seed", default=0, type=int, help="Random seed")
parser.add_argument(
    "--environment_name",
    default="LunarLander-v2",
    help="The name of the environment in the OpenAI Gym framework",
)
parser.add_argument(
    "--n_actions",
    default=4,
    type=int,
    help="The number of action that the agent can perform in the environment",
)
parser.add_argument(
    "--learning_rate",
    default="auto",
    help="The learning rate to be used for Q-learning. Default is: 'auto' (1/k)",
)
parser.add_argument(
    "--df", default=0.9, type=float, help="The discount factor used for Q-learning"
)
parser.add_argument(
    "--eps",
    default=0.05,
    type=float,
    help="Epsilon parameter for the epsilon greedy Q-learning",
)
parser.add_argument(
    "--input_space", default=8, type=int, help="Number of inputs given to the agent"
)
parser.add_argument(
    "--episodes",
    default=50,
    type=int,
    help="Number of episodes that the agent faces in the fitness evaluation phase",
)
parser.add_argument(
    "--episode_len",
    default=1000,
    type=int,
    help="The max length of an episode in timesteps",
)
parser.add_argument("--lambda_", default=30, type=int, help="Population size")
parser.add_argument(
    "--generations", default=1000, type=int, help="Number of generations"
)
parser.add_argument("--cxp", default=0.5, type=float, help="Crossover probability")
parser.add_argument("--mp", default=0.5, type=float, help="Mutation probability")
parser.add_argument(
    "--mutation",
    default="function-tools.mutUniformInt#low-0#up-40000#indpb-0.1",
    type=string_to_dict,
    help="Mutation operator. String in the format function-value#function_param_-value_1... The operators from the DEAP library can be used by setting the function to 'function-tools.<operator_name>'. Default: Uniform Int Mutation",
)
parser.add_argument(
    "--crossover",
    default="function-tools.cxOnePoint",
    type=string_to_dict,
    help="Crossover operator, see Mutation operator. Default: One point",
)
parser.add_argument(
    "--selection",
    default="function-tools.selTournament#tournsize-2",
    type=string_to_dict,
    help="Selection operator, see Mutation operator. Default: tournament of size 2",
)

parser.add_argument(
    "--genotype_len", default=100, type=int, help="Length of the fixed-length genotype"
)
parser.add_argument(
    "--low",
    default=-10,
    type=float,
    help="Lower bound for the random initialization of the leaves",
)
parser.add_argument(
    "--up",
    default=10,
    type=float,
    help="Upper bound for the random initialization of the leaves",
)
parser.add_argument(
    "--types",
    default=None,
    type=str,
    help="This string must contain the range of constants for each variable in the format '#min_0,max_0,step_0,divisor_0;...;min_n,max_n,step_n,divisor_n'. All the numbers must be integers.",
)

# Setup of the logging
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logdir = "logs/gym/{}_{}".format(
    date, "".join(np.random.choice(list(string.ascii_lowercase), size=8))
)
logfile = os.path.join(logdir, "log.txt")
os.makedirs(logdir)

args = parser.parse_args()

best = None
input_space_size = args.input_space
lr = "auto" if args.learning_rate == "auto" else float(args.learning_rate)

# Creation of an ad-hoc Leaf class
class CLeaf(Leaf):
    def __init__(self):
        super(CLeaf, self).__init__(
            args.n_actions,
            lr,
            args.df,
            args.eps,
            randInit=True,
            low=args.low,
            up=args.up,
        )


# Setup of the grammar
ORTHOGONAL_GRAMMAR = {
    "bt": ["<if>"],
    "if": ["if <condition>:{<action>}else:{<action>}"],
    "condition": [
        "_in_{0}<comp_op><const_type_{0}>".format(k) for k in range(input_space_size)
    ],
    "action": ['out=_leaf;leaf="_leaf"', "<if>"],
    "comp_op": [" < ", " > "],
}

types = (
    args.types
    if args.types is not None
    else ";".join(["0,10,1,10" for _ in range(input_space_size)])
)
types = types.replace("#", "")
assert len(types.split(";")) == input_space_size, "Expected {} types, got {}.".format(
    input_space_size, len(types.split(";"))
)

for index, type_ in enumerate(types.split(";")):
    rng = type_.split(",")
    start, stop, step, divisor = map(int, rng)
    consts_ = list(map(str, [float(c) / divisor for c in range(start, stop, step)]))
    ORTHOGONAL_GRAMMAR["const_type_{}".format(index)] = consts_

print(ORTHOGONAL_GRAMMAR)


# Seeding of the random number generators
random.seed(args.seed)
np.random.seed(args.seed)


# Log all the parameters
with open(logfile, "a") as f:
    vars_ = locals().copy()
    for k, v in vars_.items():
        f.write("{}: {}\n".format(k, v))


# Definition of the fitness evaluation function
def evaluate_fitness(fitness_function, leaf, genotype, episodes=args.episodes):
    """
    Converts a genotype to a phenotype, creates a tree based on the phenotype and evaluates the fitness of this tree.
    :param fitness_function: The fitness function that is used to calculate the fitness
    :param leaf: LeafClass that is used to build the tree
    :param genotype: Genotype to be converted for the tree
    :param episodes: Number of episodes to evaluate fitness

    Returns:
        fitness: The fitness of the tree that was constructed
    """
    phenotype, _ = GETranslator(ORTHOGONAL_GRAMMAR).genotype_to_str(genotype)
    bt = DecisionTree(phenotype, leaf)
    return fitness_function(bt, episodes)


def fitness(tree, episodes=args.episodes):
    """
    Calculates the fitness of a given DecisionTree on an environment.
    :param tree: A DecistionTree whose fitness is to be evaluated
    :param episodes: Number of episodes

    Returns:
        fitness: Fitness of the tree
        leaves: The leaves of the input tree
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    global_cumulative_rewards = []
    env = gym.make(args.environment_name)
    try:
        for iteration in range(episodes):
            env.seed(iteration)
            obs = env.reset()
            tree.new_episode()
            cumulated_reward = 0
            action = 0
            previous = None

            for t in range(args.episode_len):
                action = tree(obs)
                previous = obs[:]

                obs, rew, done, info = env.step(action)
                # env.render()
                tree.set_reward(rew)
                cumulated_reward += rew

                if done:
                    break

            tree.set_reward(rew)

            tree(obs)
            global_cumulative_rewards.append(cumulated_reward)
    except Exception as ex:
        if len(global_cumulative_rewards) == 0:
            global_cumulative_rewards = -1000
    env.close()

    fitness = (np.mean(global_cumulative_rewards),)
    leaves = tree.leaves
    return fitness, leaves


if __name__ == "__main__":

    def fit_fcn(x):
        return evaluate_fitness(fitness, CLeaf, x)

    with parallel_backend("multiprocessing"):
        pop, log, hof, best_leaves = grammatical_evolution(
            fit_fcn,
            inputs=input_space_size,
            leaf=CLeaf,
            individuals=args.lambda_,
            generations=args.generations,
            jobs=args.jobs,
            cx_prob=args.cxp,
            m_prob=args.mp,
            logfile=logfile,
            seed=args.seed,
            mutation=args.mutation,
            crossover=args.crossover,
            initial_len=args.genotype_len,
            selection=args.selection,
        )

    # Log best individual
    with open(logfile, "a") as log_:
        phenotype, _ = GETranslator(ORTHOGONAL_GRAMMAR).genotype_to_str(hof[0])
        phenotype = phenotype.replace('leaf="_leaf"', "")

        for k in range(50000):  # Iterate over all possible leaves
            key = "leaf_{}".format(k)
            if key in best_leaves:
                v = best_leaves[key].q
                phenotype = phenotype.replace(
                    "out=_leaf", "out={}".format(np.argmax(v)), 1
                )
            else:
                break

        log_.write(str(log) + "\n")
        log_.write(str(hof[0]) + "\n")
        log_.write(phenotype + "\n")
        log_.write("best_fitness: {}".format(hof[0].fitness.values[0]))
    with open(os.path.join(logdir, "fitness.tsv"), "w") as f:
        f.write(str(log))
