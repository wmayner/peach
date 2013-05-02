from __future__ import division # makes 1/2 equal float 0.5 and not integer 0
import random

BEST_K   = None
BEST_EPS = None

class Rmax:
  def __init__(self, rmax, gamma, K, actions, iters=100):
    self.actions = actions

  def update(self, s,a,r,s_):
    pass
  def get_action(self, s):
    return random.choice(self.actions)

class Qlearner:
  def __init__(self,alpha, gamma, actions, epsilon):
    self.actions = actions

  def update(self, s,a,r,s_):
    pass
  def get_action(self, s):
    return random.choice(self.actions)
