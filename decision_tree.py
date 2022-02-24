#!/usr/bin/python3
"""
Implementation of the class  DecisionTree and the class Leaf 
incl. q-learning and random init.

"""
import numpy as np
import abc

class DecisionTree:
  def __init__(self):
    self.current_reward = 0
    self.last_leaf = None

  @abc.abstractmethod
  def get_action(self, input):
    pass

  def set_reward(self, reward):
    self.current_reward = reward

  def reset(self):
    self.last_leaf = None


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
      leaf.q[leaf.last_action] += leaf.learning_rate * (reward + leaf.discount_factor * q_next - leaf.q[leaf.last_action])

  def next_iteration(self):
    self.iteration[self.last_action] += 1

  def __repr__(self):
      return ", ".join(["{:.2f}".format(k) for k in self.q])

  def __str__(self):
      return repr(self)
