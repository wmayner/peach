from __future__ import division # makes 1/2 equal float 0.5 and not integer 0
import random

BEST_K   = None
BEST_EPS = None

class Rmax:
  def __init__(self, rmax, gamma, K, actions, iters=100):
    self.actions, self.rmax, self.gamma, self.iters, self.K = actions, rmax, gamma, iters, K
    self.C, self.R, self.T, self.V, self.S = {},{},{},{},set()
    self.needs_updating = False
    self.epsilon = 0.1

  def valiter(self, S, T, C, R, gamma, epsilon):
    U = dict((s,0) for s in S)
    U_ = U
    delta = float('inf')
    while delta >= epsilon*(1-gamma)/gamma:
      # print "---Valiter---"
      U, delta = U_, 0
      for s in S:
        max_terms = []
        for a in self.actions:
          summation_terms = []
          for s_ in S:
            if C[s,a] == 0:
              summation_terms += [0]
            else:
              summation_terms += [ (T[s,a,s_] / C[s,a]) * U[s_] ]
          # print summation_terms
          max_terms += [sum(summation_terms)]
        # print "maxterms:",max_terms
        # print "max of terms:",max(max_terms)
        U_[s] = R[s] + max(max_terms) + random.random()/10000
        newdiff = abs(U_[s] - U[s])
        # print newdiff
        if newdiff > delta:
          delta = newdiff
    # print "Valiter:"
    # print U
    # print ""
    return U

  def update(self, s,a,r,s_):

    # initialize/update T
    if (s,a,s_) not in self.T:
      self.T[s,a,s_] = 1
    else:
      self.T[s,a,s_] += 1
    # run value iteration if transition probabilities have changed
    # i.e. if there's more than one state that (s,a) leads to
    if len([state for state in self.S if self.T[s,a,state] > 0]) > 1:
      self.needs_updating = True
    # update R
    self.R[s_] = r
    # initialize V if necessary

    # run Value Iteration if we need to
    if self.needs_updating or self.C[s,a] == self.K:
      self.V = self.valiter(self.S, self.T, self.C, self.R, self.gamma, self.epsilon)

  def get_action(self, s):

    if s not in self.S:
      self.needs_updating = True
      # record that we've seen s
      self.S.add(s)
      # initialize V
      self.V[s] = 0
      # initialize R
      self.R[s] = 0
      # initialize C and T
      for a in self.actions:
        self.C[s,a] = 0
        for s_ in self.S:
          self.T[s_,a,s] = 0
          self.T[s,a,s_] = 0

    # continue exploring if C[s,a] < K
    utopian_actions = [a for a in self.actions if self.C[s,a] < self.K]
    # for debug in ["C["+str( s )+","+str(action)+"]: "+str(self.C[s,action]) for action in self.actions]:
    #   print debug
    if utopian_actions:
      # update C
      choice = random.choice(utopian_actions)
      self.C[s,choice] += 1
      return choice

    # otherwise, begin exploiting
    action_values = {}
    for a in self.actions:
      reachable_states = [s_ for state,action,s_ in self.T if (state,action) == (s,a)]
      for s_ in reachable_states:
        if s_ not in self.V:
          self.V[s_] = 0
        # print s_
        # print self.V[s_]
        # print self.T[s,a,s_]
        # print self.C[s,a]
      action_values[a] = sum(self.V[s_] * self.T[s,a,s_]/self.C[s,a] for s_ in reachable_states)
    # print "Best actions:"
    # print action_values
    best_action = sorted(action_values.keys(), key=action_values.get)[-1]
    utopian_actions = [a for a in self.actions if self.C[s,a] < self.K]

    # update C
    self.C[s, best_action] += 1

    return best_action

class Qlearner:
  def __init__(self,alpha, gamma, actions, epsilon):
    self.alpha = alpha
    self.gamma = gamma
    self.actions = actions
    self.Q = {}
    self.epsilon = 0.10

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
    Q = self.Q
    if s not in Q or random.random() < self.epsilon:
      return random.choice(self.actions)
    return sorted(Q[s].keys(), key=Q[s].get)[-1]
