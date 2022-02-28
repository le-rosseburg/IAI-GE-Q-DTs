import os
import gym
import string
import datetime
import argparse
import numpy as np
import stopit
from time import time, sleep
from numpy import random
from matplotlib import pyplot as plt
from multiprocessing import TimeoutError

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
parser.add_argument("--population_size", default=30, type=int, help="Population size")
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
    "--genotype_len", default=1024, type=int, help="Length of the fixed-length genotype"
)  # default for oblique 100
parser.add_argument(
    "--low",
    default=-10,
    type=float,
    help="Lower bound for the random initialization of the leaves",
)  # Not used in oblique in original
parser.add_argument(
    "--up",
    default=10,
    type=float,
    help="Upper bound for the random initialization of the leaves",
)  # Not used in oblique in original

parser.add_argument(
    "--decay",
    default=0.99,
    type=float,
    help="The decay factor for the epsilon decay (eps_t = eps_0 * decay^t)",
)  # Not used in orthogonal  in original
parser.add_argument(
    "--patience",
    default=50,
    type=int,
    help="Number of episodes to use as evaluation period for the early stopping",
)  # Not used in orthogonal  in original
parser.add_argument(
    "--timeout",
    default=600,
    type=int,
    help="Maximum evaluation time, useful to continue the evolution in case of MemoryErrors",
)  # Not used in orthogonal in original
parser.add_argument(
    "--with_bias",
    action="store_true",
    help="if used, then the the condition will be (sum ...) < <const>, otherwise (sum ...) < 0",
)  # Not used in orthogonal in original
parser.add_argument(
    "--random_init",
    action="store_true",
    help="Randomly initializes the leaves in [-1, 1[",
)  # Not used in orthogonal in original
parser.add_argument(
    "--constant_range",
    default=1000,
    type=int,
    help="Max magnitude for the constants being used (multiplied *10^-3). Default: 1000 => constants in [-1, 1]",
)  # Not used in orthogonal
parser.add_argument(
    "--constant_step",
    default=1,
    type=int,
    help="Step used to generate the range of constants, mutliplied *10^-3",
)  # Not used in orthogonal
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
fitfile = os.path.join(logdir, "fitness.tsv")
pltfile_jpg = os.path.join(logdir, "fitness.jpg")
pltfile_pdf = os.path.join(logdir, "fitness.pdf")
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


# Creation of the EpsilonDecay Leaf
class EpsilonDecayLeaf(Leaf):
    """A eps-greedy leaf with epsilon decay."""

    def __init__(self):
        """
        Initializes the leaf
        """
        if not args.random_init:
            Leaf.__init__(
                self,
                n_actions=args.n_actions,
                learning_rate=lr,
                discount_factor=args.df,
                epsilon=args.eps,
                low=0,
                up=0,
            )
        else:
            Leaf.__init__(
                self,
                n_actions=args.n_actions,
                learning_rate=lr,
                discount_factor=args.df,
                epsilon=args.eps,
                low=-1,
                up=1,
            )

        self._decay = args.decay
        self._steps = 0

    def get_action(self):
        self.epsilon = self.epsilon * self._decay
        self._steps += 1
        return super().get_action()


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

consts = {}
for index, type_ in enumerate(types.split(";")):
    rng = type_.split(",")
    start, stop, step, divisor = map(int, rng)
    consts_ = list(map(str, [float(c) / divisor for c in range(start, stop, step)]))
    consts[index] = (consts_[0], consts_[-1])
    ORTHOGONAL_GRAMMAR["const_type_{}".format(index)] = consts_

# print(ORTHOGONAL_GRAMMAR)
oblique_split = "+".join(
    [
        "<const> * (_in_{0} - {1})/({2} - {1})".format(i, consts[i][0], consts[i][1])
        for i in range(input_space_size)
    ]
)

OBLIQUE_GRAMMAR = {
    "bt": ["<if>"],
    "if": ["if <condition>:{<action>}else:{<action>}"],
    "action": ['out=_leaf;leaf="_leaf"', "<if>"],
    # "const": ["0", "<nz_const>"],
    "const": [
        str(k / 1000)
        for k in range(
            -args.constant_range, args.constant_range + 1, args.constant_step
        )
    ],
}

if not args.with_bias:
    OBLIQUE_GRAMMAR["condition"] = [oblique_split + " < 0"]
else:
    OBLIQUE_GRAMMAR["condition"] = [oblique_split + " < <const>"]


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


# Maybe combine with orthogonal
def evaluate_fitness_oblique(
    fitness_function_oblique, leaf, genotype, episodes=args.episodes
):
    repeatable_random_seed = sum(genotype) % (2**31)
    random.seed(args.seed + repeatable_random_seed)
    np.random.seed(args.seed + repeatable_random_seed)
    phenotype, _ = GETranslator(OBLIQUE_GRAMMAR).genotype_to_str(genotype)
    bt = DecisionTree(phenotype, leaf)
    return fitness_function_oblique(bt, episodes, timeout=args.timeout)


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
                action = tree.get_action(obs)
                previous = obs[:]

                obs, rew, done, info = env.step(action)
                # env.render()
                tree.set_reward(rew)
                cumulated_reward += rew

                if done:
                    break

            tree.set_reward(rew)

            tree.get_action(obs)
            global_cumulative_rewards.append(cumulated_reward)
    except Exception as ex:
        if len(global_cumulative_rewards) == 0:
            global_cumulative_rewards = -1000
    env.close()

    fitness = (np.mean(global_cumulative_rewards),)
    leaves = tree.leaves
    return fitness, leaves


# Combine with orthogonal
@stopit.threading_timeoutable(default=((-1000,), None))
def fitness_oblique(tree, episodes=args.episodes):
    random.seed(args.seed)
    np.random.seed(args.seed)
    global_cumulative_rewards = []
    env = gym.make(args.environment_name)
    initial_perf = None
    try:
        for iteration in range(episodes):
            env.seed(iteration)
            obs = env.reset()
            tree.new_episode()
            cumulated_reward = 0
            action = 0
            previous = None

            for t in range(args.episode_len):
                obs = list(obs.flatten())
                action = tree(obs)
                previous = obs[:]
                obs, rew, done, _ = env.step(action)
                tree.set_reward(rew)
                cumulated_reward += rew
                if done:
                    break

            tree(obs)
            global_cumulative_rewards.append(cumulated_reward)

            # Check stopping criterion
            if initial_perf is None and iteration >= args.patience:
                initial_perf = np.mean(global_cumulative_rewards)
            elif iteration % args.patience == 0 and iteration > args.patience:
                if (
                    np.mean(global_cumulative_rewards[-args.patience :]) - initial_perf
                    < 0
                ):
                    break
                initial_perf = np.mean(global_cumulative_rewards[-args.patience :])
    except Exception as ex:
        if len(global_cumulative_rewards) == 0:
            global_cumulative_rewards = [-1000]
    env.close()

    fitness = (np.mean(global_cumulative_rewards[-args.patience :]),)
    return fitness, tree.leaves


if __name__ == "__main__":
    # Check which grammar is used to create/use the appriopriate method
    def fit_fcn(tree):
        return evaluate_fitness(fitness, CLeaf, tree)

    def fit_fcn_oblique(tree):
        return evaluate_fitness(fitness_oblique, EpsilonDecayLeaf, tree)

    with parallel_backend("multiprocessing"):
        pop, log, hof, best_leaves = grammatical_evolution(
            fitness_function=fit_fcn,
            n_individuals=args.population_size,
            n_generations=args.generations,
            jobs=args.jobs,
            cxpb=args.cxp,
            mutpb=args.mp,
            genotype_len=args.genotype_len,
            selection=args.selection,
            mutation=args.mutation,
            crossover=args.crossover,
            seed=args.seed,
            logfile=logfile,
        )  # timeout and initional_len missing for oblique as well as eval() in
        # the beginning, (leaf, fitfcn depend on grammar)

    # Log fitness inside .tsv-file
    with open(fitfile, "a") as fit_:
        fit_.write(str(log))

    # Log best individual
    with open(logfile, "a") as log_:
        phenotype, _ = GETranslator(ORTHOGONAL_GRAMMAR).genotype_to_str(
            hof[0]
        )  ### Change to used grammar
        phenotype = phenotype.replace('leaf="_leaf"', "")

        # Iterate over all possible leaves
        for k in range(50000):
            key = "leaf_{}".format(k)
            if key in best_leaves:
                v = best_leaves[key].q
                phenotype = phenotype.replace(
                    "out=_leaf", "out={}".format(np.argmax(v)), 1
                )
            else:
                break

        log_.write("\n" + "Fitness history:\n" + str(log) + "\n")
        log_.write("\n" + "HOF-Individual:\n" + str(hof[0]) + "\n")
        log_.write("\n" + "Phenotype:\n" + phenotype + "\n")
        log_.write("best_fitness: {}".format(hof[0].fitness.values[0]))

    # Plotting result
    plt.title(args.environment_name)
    plt.xlabel("generations")
    plt.ylabel("fitness score")
    plt.xlim(-2, args.generations + 2)
    xpoints = []
    minpoints = []
    maxpoints = []
    avgpoints = []
    stdpoints = []
    for i in range(0, len(log)):
        xpoints.append(log[i]["gen"])
        maxpoints.append(log[i]["max"])
        minpoints.append(log[i]["min"])
        avgpoints.append(log[i]["avg"])
        stdpoints.append(log[i]["std"])
    plt.ylim(-10, max(maxpoints) + 10)
    plt.plot(xpoints, maxpoints, label="max", color="#2ca02c")
    plt.plot(xpoints, minpoints, label="min", color="#ff7f0e")
    plt.plot(xpoints, avgpoints, label="avg", color="#1f77b4")
    # add errorbars
    stdtop = []
    stdbottom = []
    for i in range(0, len(log)):
        stdtop.append(avgpoints[i] + stdpoints[i] / 2)
        stdbottom.append(avgpoints[i] - stdpoints[i] / 2)
    plt.fill_between(xpoints, stdtop, stdbottom, color="#86c1ea")
    plt.legend()
    plt.savefig(pltfile_jpg, format="jpg")
    plt.savefig(pltfile_pdf, format="pdf")
    plt.show()
