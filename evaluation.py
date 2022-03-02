"""
In this file one can evaluate programs on OpenAI Gym tasks.

"""
import gym
import argparse
import datetime
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--grammar", default="orthogonal", type=str, help="The grammar that will be used"
)
parser.add_argument(
    "--environment",
    default="CartPole-v1",
    help="The name of the environment in the OpenAI Gym framework",
)
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--n_runs", default=100, type=int, help="Number of runs")


def program(input_, environment, grammar):
    if environment == "CartPole-v1":
        if grammar == "orthogonal":
            # seed 7 result
            if input_[3] > 0.509:
                out = 1
            else:
                if input_[2] > 0.022:
                    out = 1
                else:
                    out = 0
        if grammar == "oblique":
            # seed 19 result
            if (
                0.516 * input_[0]
                + 0.808 * input_[1]
                + 0.05 * input_[2]
                + -0.727 * input_[3]
                < 0.21
            ):
                out = 1
            else:
                out = 0
    elif environment == "MountainCar-v0":
        if grammar == "orthogonal":
            # seed 7 result
            if input_[1] > 0.015:
                out = 2
            else:
                if input_[1] < -0.06:
                    out = 2
                else:
                    if input_[0] > 0.25:
                        if input_[1] < -0.035:
                            out = 1
                        else:
                            if input_[0] > 0.15:
                                out = 2
                            else:
                                out = 1
                    else:
                        if input_[0] > -0.9:
                            if input_[1] < 0.0:
                                out = 0
                            else:
                                if input_[0] < -0.45:
                                    if input_[0] > -0.1:
                                        out = 2
                                    else:
                                        out = 2
                                else:
                                    if input_[1] < -0.07:
                                        out = 1
                                    else:
                                        if input_[0] < -0.45:
                                            if input_[0] > 0.4:
                                                out = 1
                                            else:
                                                out = 2
                                        else:
                                            out = 0
                        else:
                            if input_[1] < -0.025:
                                out = 0
                            else:
                                if input_[1] > 0.06:
                                    out = 1
                                else:
                                    out = 2
        if grammar == "oblique":
            # seed x result
            out = 0
    elif environment == "CartPole-v1":
        if grammar == "orthogonal":
            # seed x result
            out = 0
        if grammar == "oblique":
            # seed x result
            out = 0
    return out


def evaluate(program, environment, grammar, seed):
    if environment == "CartPole-v1":
        env = gym.make("CartPole-v1")
    elif environment == "MountainCar-v0":
        env = gym.make("MountainCar-v1")
    elif environment == "LunarLander-v2":
        env = gym.make("LunarLander-v2")

    env.seed(seed)
    obs = env.reset()

    cumulated_reward = 0
    time = datetime.datetime.now().time()

    while True:

        action = program(obs)
        obs, rew, done, _ = env.step(action)
        cumulated_reward += rew

        if done:
            break

    env.close()

    print("Evaluation took {}s".format(datetime.datetime.now().time() - time))
    print("Score: {}".format(cumulated_reward))
    return cumulated_reward


if __name__ == "__main__":

    scores = [
        evaluate(program, args.environment, args.grammar, s)
        for s in range(args.seed, args.seed + args.n_runs)
    ]
    print("Mean: {}; Std: {}".format(np.mean(scores), np.std(scores)))
