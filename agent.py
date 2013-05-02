from __future__ import division # makes 1/2 equal float 0.5 and not integer 0
import random
import operator

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
    self.Q = {}

  def update(self, s,a,r,s_):
    gamma = self.gamma
    alpha = self.alpha
    actions = self.actions
    Q = self.Q

    # initialize Q if necessary
    for state in [st for st in [s,s_] if st not in Q]:
      Q[state] = dict((a,0) for a in actions)

    # compute update step
    Q[s][a] = (1 - alpha) * Q[s][a] + \
      alpha * (r + gamma * max(Q[s_][a_] for a_ in actions))

  def get_action(self, s):
    return sorted(Q[s].keys(), key=operator.itemgetter(1))[0]
