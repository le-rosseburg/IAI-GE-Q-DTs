#!/usr/bin/python3
"""
Implementation of the class  DecisionTree and the class Leaf 
incl. q-learning and random init.

"""
import numpy as np

class DecisionTree:
  def __init__(self, phenotype, leaf):
    self.program = phenotype
    self.current_reward = 0
    self.leaves = {}
    self.leaf_count = 0
    self.last_leaf = None

    while "_leaf" in self.program:
      # create new leaf, add leaf to dictionary and update the leaf counter
      new_leaf = leaf()
      leaf_name = "leaf_{}".format(self.leaf_count)
      self.leaves[leaf_name] = new_leaf
      self.leaf_count += 1

      # replace phentype entry with newly created leaf
      self.program = self.program.replace("_leaf", "'{}.get_action()'".format(leaf_name), 1)
      self.program = self.program.replace("_leaf", "{}".format(leaf_name), 1)

    self.exec_ = compile(self.program, "<string>", "exec", optimize=2)

  def get_action(self, input):
    if len(self.program) == 0:
        return None

    """what exactly is the input? how does it look?"""
    variables = {} # {"out": None, "leaf": None}
    for idx, i in enumerate(input):
      variables["_in_{}".format(idx)] = i
    variables.update(self.leaves)

    """what exactly is executed? how does the phenotype look and what is done with phentype(string) + variables(dict)"""
    exec(self.exec_, variables)

    current_leaf = self.leaves[variables["leaf"]]
    current_q_value = max(current_leaf.q)
    if self.last_leaf is not None:
        # apply q-leraning update
        self.last_leaf.update(self.current_reward, current_q_value)

    # update last used leaf
    self.last_leaf = current_leaf 
    return current_leaf.get_action()

  def set_reward(self, reward):
    self.current_reward = reward

  def new_episode(self):
    self.last_leaf = None

  def __call__(self, x):
    return self.get_action(x)

  def __str__(self):
    return self.program


class Leaf:
  def __init__(self, n_actions, learning_rate, discount_factor, epsilon, randInit=False, low=None, up=None):
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor
    self.epsilon = epsilon
    self.parent = None
    self.n_actions = n_actions
    self.last_action = None
    self.used_actions = [1] * n_actions

    # check if the q-values of the leaf should be random initialized or not
    if randInit:
      self.q = np.random.uniform(low, up, self.n_actions, dtype=np.float32)
    else:
      self.q = np.zeros(self.n_actions, dtype=np.float32)

  def get_action(self):
    # apply e-greedy strategy
    if np.random.uniform() < self.epsilon:
      # choose random action
      action = self.n_actions.sample()
    else:
      # choose randomly between all actions that have the maximum q-value
      max_v = max(self.q)
      indices = [i for i, v in enumerate(self.q) if v == max_v]
      action = self.q[np.random.choice(indices)]

    # update and count last used action
    self.last_action = action
    self.count_actions()
    return action

  def update(self, reward, q_next):
    if self.last_action is not None:
      """still don't know why they check if self-learning_rate is callable"""
      # Special szenario, used for LunarLander environment:
      # "The learning rate has been set to 1/k, where k is the number of visits to the [last] state-action pair."
      # "This guarantees that the state-action function converges to the optimum with k → ∞"
      if lr == "auto":
        lr = 1/self.used_actions[self.last_action]
      # apply q-learning update
      self.q[self.last_action] += self.learning_rate * (reward + self.discount_factor * q_next - self.q[self.last_action])

  def count_actions(self):
    # count the number of times each action is used within the iteration
    self.used_actions[self.last_action] += 1

  def __repr__(self):
      return ", ".join(["{:.2f}".format(k) for k in self.q])

  def __str__(self):
      return repr(self)
