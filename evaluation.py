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
parser.add_argument("--seed", default=42, type=int, help="Seed used for training")
parser.add_argument("--n_runs", default=100, type=int, help="Number of runs")
args = parser.parse_args()


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
            # seed 62 result
            if (
                -0.012 * input_[0]
                + -0.355 * input_[1]
                + -0.205 * input_[2]
                + -0.934 * input_[3]
                < 0.072
            ):
                if (
                    -0.399 * input_[0]
                    + 0.799 * input_[1]
                    + 0.508 * input_[2]
                    + 0.161 * input_[3]
                    < 0.319
                ):
                    out = 1
                else:
                    if (
                        -0.821 * input_[0]
                        + -0.862 * input_[1]
                        + -0.687 * input_[2]
                        + 0.037 * input_[3]
                        < 0.839
                    ):
                        if (
                            0.939 * input_[0]
                            + 0.856 * input_[1]
                            + -0.927 * input_[2]
                            + -0.593 * input_[3]
                            < -0.453
                        ):
                            out = 1
                        else:
                            if (
                                -0.482 * input_[0]
                                + -0.508 * input_[1]
                                + 0.718 * input_[2]
                                + -0.272 * input_[3]
                                < 0.363
                            ):
                                if (
                                    0.496 * input_[0]
                                    + -0.532 * input_[1]
                                    + 0.355 * input_[2]
                                    + 0.069 * input_[3]
                                    < 0.096
                                ):
                                    out = 1
                                else:
                                    out = 0
                            else:
                                out = 0
                    else:
                        out = 0
            else:
                if (
                    0.339 * input_[0]
                    + 0.648 * input_[1]
                    + 0.277 * input_[2]
                    + 0.88 * input_[3]
                    < 0.699
                ):
                    out = 0
                else:
                    if (
                        -0.841 * input_[0]
                        + 0.324 * input_[1]
                        + 0.603 * input_[2]
                        + -0.45 * input_[3]
                        < 0.83
                    ):
                        out = 0
                    else:
                        if (
                            0.4 * input_[0]
                            + -0.09 * input_[1]
                            + -0.007 * input_[2]
                            + -0.047 * input_[3]
                            < -0.356
                        ):
                            out = 0
                        else:
                            out = 1
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
            # seed 7 result
            if (
                -0.603 * (input_[0] - -1.2) / (0.65 - -1.2)
                + 0.111 * (input_[1] - -0.07) / (0.065 - -0.07)
                < -0.294
            ):
                out = 2
            else:
                if (
                    -0.757 * (input_[0] - -1.2) / (0.65 - -1.2)
                    + 0.866 * (input_[1] - -0.07) / (0.065 - -0.07)
                    < 0.148
                ):
                    if (
                        0.642 * (input_[0] - -1.2) / (0.65 - -1.2)
                        + -0.778 * (input_[1] - -0.07) / (0.065 - -0.07)
                        < 0.358
                    ):
                        out = 0
                    else:
                        if (
                            0.278 * (input_[0] - -1.2) / (0.65 - -1.2)
                            + 0.041 * (input_[1] - -0.07) / (0.065 - -0.07)
                            < 0.389
                        ):
                            if (
                                0.932 * (input_[0] - -1.2) / (0.65 - -1.2)
                                + -0.906 * (input_[1] - -0.07) / (0.065 - -0.07)
                                < -0.1
                            ):
                                out = 1
                            else:
                                out = 0
                        else:
                            out = 2
                else:
                    if (
                        -0.479 * (input_[0] - -1.2) / (0.65 - -1.2)
                        + 0.69 * (input_[1] - -0.07) / (0.065 - -0.07)
                        < 0.226
                    ):
                        if (
                            -0.106 * (input_[0] - -1.2) / (0.65 - -1.2)
                            + -0.173 * (input_[1] - -0.07) / (0.065 - -0.07)
                            < -0.119
                        ):
                            out = 2
                        else:
                            if (
                                0.709 * (input_[0] - -1.2) / (0.65 - -1.2)
                                + 0.69 * (input_[1] - -0.07) / (0.065 - -0.07)
                                < 0.371
                            ):
                                out = 1
                            else:
                                out = 0
                    else:
                        out = 2
    elif environment == "LunarLander-v2":
        if grammar == "oblique":
            # seed x result
            out = 0
    return out


def evaluate(program, environment, grammar, seed):
    if environment == "CartPole-v1":
        env = gym.make("CartPole-v1")
    elif environment == "MountainCar-v0":
        env = gym.make("MountainCar-v0")
    elif environment == "LunarLander-v2":
        env = gym.make("LunarLander-v2")

    env.seed(seed)
    obs = env.reset()
    cumulated_reward = 0

    while True:

        action = program(obs, environment, grammar)
        obs, rew, done, _ = env.step(action)
        cumulated_reward += rew

        if done:
            break

    env.close()
    return cumulated_reward


if __name__ == "__main__":

    eval_path = "Results/{}/{}/eval.txt".format(args.environment, args.grammar)

    start_time = datetime.datetime.now()
    scores = [
        evaluate(program, args.environment, args.grammar, seed)
        for seed in range(args.seed + 1, args.seed + args.n_runs + 1)
    ]
    end_time = datetime.datetime.now()

    with open(eval_path, "a") as eval_:
        eval_.write("Environment: " + args.environment + "; ")
        eval_.write("Grammar: " + args.grammar + "; ")
        eval_.write("Seed: {}; ".format(args.seed))
        eval_.write("Number of runs: {};\n".format(args.n_runs))
        eval_.write(
            "Mean = {} \nStd = {}".format(np.mean(scores), np.std(scores)) + "\n"
        )
        eval_.write("Start time: {}\n".format(start_time))
        eval_.write("End time: {}\n\n".format(end_time))
