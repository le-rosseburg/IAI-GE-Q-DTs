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
      new_leaf = leaf()
      leaf_name = "leaf_{}".format(self.leaf_count)
      self.leaves[leaf_name] = new_leaf
      self.leaf_count += 1

      self.program = self.program.replace("_leaf", "'{}.get_action()'".format(leaf_name), 1)
      self.program = self.program.replace("_leaf", "{}".format(leaf_name), 1)

    self.exec_ = compile(self.program, "<string>", "exec", optimize=2)

  def get_action(self, input):
    if len(self.program) == 0:
        return None
    variables = {} # {"out": None, "leaf": None}
    for idx, i in enumerate(input):
      variables["_in_{}".format(idx)] = i
    variables.update(self.leaves)

    exec(self.exec_, variables)

    current_leaf = self.leaves[variables["leaf"]]
    current_q_value = max(current_leaf.q)
    if self.last_leaf is not None:
        self.last_leaf.update(self.current_reward, current_q_value)
    self.last_leaf = current_leaf 
    
    return current_leaf.get_action()
  
  def set_reward(self, reward):
    self.current_reward = reward

  def reset(self):
    self.last_leaf = None

  def __call__(self, x):
    return self.get_action(x)

  def __str__(self):
    return self.program


class Leaf:
  def __init__(self, n_actions, learning_rate, discount_factor, epsilon, randInit=False, low=None, up=None):
    self.n_actions = n_actions
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor
    self.epsilon = epsilon
    self.parent = None
    self.last_action = None
    self.iteration = [1] * n_actions
    # check if random initialization
    if randInit:
      self.q = np.random.uniform(low, up, self.n_actions, dtype=np.float32)
    else:
      self.q = np.zeros(self.n_actions, dtype=np.float32)
  
  def get_action(self):
    # get e-greedy action
    if np.random.uniform() < self.epsilon:
      action = self.n_actions.sample()
    else:
      # choose randomly between all argmax
      max_v = max(self.q)
      indices = [i for i, v in enumerate(self.q) if v == max_v]
      action = self.q[np.random.choice(indices)]
    
    self.last_action = action
    self.next_iteration()
    return action
  
  def update(self, reward, q_next):
    if self.last_action is not None:
      self.q[self.last_action] += self.learning_rate * (reward + self.discount_factor * q_next - self.q[self.last_action])

  def next_iteration(self):
    self.iteration[self.last_action] += 1

  def __repr__(self):
      return ", ".join(["{:.2f}".format(k) for k in self.q])

  def __str__(self):
      return repr(self)
