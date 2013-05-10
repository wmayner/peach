from __future__ import division # makes 1/2 equal float 0.5 and not integer 0
import random

BEST_K   = 3
BEST_EPS = 0.5

class Rmax:
  def __init__(self, rmax, gamma, K, actions, iters=100):
    self.actions, self.rmax, self.gamma, self.iters, self.K = actions, rmax, gamma, iters, K
    self.C, self.R, self.T, self.V, self.S = {},{},{},{},set()
    self.needs_updating = False
    self.epsilon = 1
    self.update_count = 0


  def valiter(self, S, T, C, R, gamma, epsilon):
    U_ = dict((s,random.random()) for s in S)
    delta = float('inf')
    while delta >= epsilon*(1-gamma)/gamma:
      U, delta = dict(U_), 0
      for s in S:
        # max_terms = []
        # for a in self.actions:
        #   summation_terms = []
        #   for s_ in S:
        #     if C[s,a] == 0:
        #       summation_terms += [0]
        #     else:
        #       summation_terms += [ (T[s,a,s_] / C[s,a]) * U[s_] ]
        #   # print summation_terms
        #   max_terms += [sum(summation_terms)]
        # print "maxterms:",max_terms
        # print "max of terms:",max(max_terms)
        # U_[s] = R[s] + gamma * max(max_terms) + random.random()/10000
        max_args = [ R[s,a] / C[s,a] for a in self.actions if (s,a) in R]
        max_term = max(max_args) if max_args else 0
        U_[s] = max_term + gamma * \
            max( sum( 0 if C[s,a] == 0 else (T[s,a,s_] / C[s,a]) * U[s_] for s_ in S) for a in self.actions)
        newdiff = abs(U_[s] - U[s])
        if newdiff > delta:
          delta = newdiff
    return U

  def record_new_state(self, s):

    def init_C(new_state):
      for action in self.actions:
        if (new_state, action) not in self.C:
          self.C[new_state, action] = 0

    def init_T(new_state):
      for old_state in self.S:
        for action in self.actions:
          if (new_state, action, old_state) not in self.T:
            self.T[new_state, action, old_state] = 0
          if (old_state, action, new_state) not in self.T:
            self.T[old_state, action, new_state] = 0

    # record that we've seen it
    self.S.add(s)
    # re-run value iteration
    self.needs_updating = True
    # initialize C
    init_C(s)
    # initialize T
    init_T(s)


  def update(self, s,a,r,s_):
    # if self.update_count % 10 == 0:
    #   self.needs_updating = True

    # if the resulting state is new, initialize and re-run valiter
    if s_ not in self.S:
      self.record_new_state(s_)

    # update T
    self.T[s,a,s_] += 1

    # update R
    if (s,a) not in self.R:
      self.R[s,a] = r
    else:
      self.R[s,a] += r

    # run Value Iteration if:
    # - a new state has been observed
    # - a state-action pair has been attempted K times
    # - transition probabilities have changed, i.e. if there's more than one
    #   state that (s,a) leads to
    if self.needs_updating \
       or self.C[s,a] == self.K \
       or len([state for state in self.S if self.T[s,a,state] > 0]) > 1:
      self.V = self.valiter(self.S, self.T, self.C, self.R, self.gamma, self.epsilon)
      self.needs_updating = False


  def get_action(self, s):
    # if the resulting state is new, initialize and rerun valiter
    if s not in self.S:
      self.record_new_state(s)

    # otherwise, begin exploiting
    action_values = {}
    for a in self.actions:
      reachable_states = [s_ for state,action,s_ in self.T if (state,action) == (s,a)]
      if self.C[s,a] < self.K:
        action_values[a] = self.rmax
      else:
        action_values[a] = sum(self.V[s_] * self.T[s,a,s_] / self.C[s,a] for s_ in reachable_states)

    # print "C[",s,"]:", [self.C[s,a] for a in self.actions]
    # print sorted(action_values.values())
    top_value = sorted(action_values.values())[-1]
    # print "Top value:", top_value
    best_actions = [a for a,v in action_values.iteritems() if v == top_value]
    # print "Best_actions:",best_actions
    choice = random.choice(best_actions)
    # print choice

    if self.C[s,choice] > 150:
      choice = random.choice(self.actions)

    # update C
    self.C[s, choice] += 1
    return choice


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
