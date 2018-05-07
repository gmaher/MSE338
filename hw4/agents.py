# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:28:12 2018

@author: Maria Dimakopoulou
"""

import numpy as np

###############################################################################
class Agent(object):
  """Base class for all agent interface."""

  def __init__(self, **kwargs):
    pass

  def __str__(self):
    pass

  def update_observation(self, obs, action, reward, new_obs, p_continue,
                         **kwargs):
    pass

  def update_policy(self, **kwargs):
    pass

  def pick_action(self, obs, **kwargs):
    pass

  def initialize_episode(self, **kwargs):
    pass

  def _random_argmax(self, vector):
    """Helper function to select argmax at random... not just first one."""
    # TODO(Implement _random_argmax)
    pass

  def _egreedy_action(self, q_vals, epsilon):
    """Epsilon-greedy dithering action selection.
    Args:
      q_vals: n_action x 1 - Q value estimates of each action.
      epsilon: float - probability of random action
    Returns:
      action: integer index for action selection
    """
    # TODO(Implement _egreedy_action)
    pass


  def _boltzmann_action(self, q_vals, beta):
    """Boltzmann dithering action selection.
    Args:
      q_vals: n_action x 1 - Q value estimates of each action.
      beta: float - temperature for Boltzmann
    Returns:
      action - integer index for action selection
    """
    # TODO(Implement _boltzmann_action)
    pass


class RandomAgent(Agent):
  """Take actions completely at random."""
  def __init__(self, num_action, feature_extractor, **kwargs):
    self.num_action = num_action
    self.feature_extractor = feature_extractor

  def __str__(self):
    return "RandomAgent(|A|={})".format(self.num_action)

  def pick_action(self, obs, **kwargs):
    state = self.feature_extractor.get_feature(obs)
    action = np.random.randint(self.num_action)
    return action


class ConstantAgent(Agent):
  """Take constant actions."""
  def __init__(self, action, feature_extractor, **kwargs):
    self.action = action
    self.feature_extractor = feature_extractor

  def __str__(self):
    return "ConstantAgent(a={})".format(self.action)

  def pick_action(self, obs, **kwargs):
    state = self.feature_extractor.get_feature(obs)
    return self.action


# TODO(Implement EpisodicQLearning)
# class EpisodicQLearning(Agent):


# TODO(Implement SARSA)
# class SARSA(Agent):


###############################################################################
class FeatureExtractor(object):
  """Base feature extractor."""

  def __init__(self, **kwargs):
    pass

  def __str__(self):
    pass

  def get_feature(self, obs):
    pass


class TabularFeatures(FeatureExtractor):

  def __init__(self, num_x, num_x_dot, num_theta, num_theta_dot):
    """Define buckets across each variable."""
    self.num_x = num_x
    self.num_x_dot = num_x_dot
    self.num_theta = num_theta
    self.num_theta_dot = num_theta_dot

    self.x_bins = np.linspace(-3, 3, num_x - 1, endpoint=False)
    self.x_dot_bins = np.linspace(-2, 2, num_x_dot - 1, endpoint=False)
    self.theta_bins = np.linspace(- np.pi / 3, np.pi / 3,
                                  num_theta - 1, endpoint=False)
    self.theta_dot_bins = np.linspace(-4, 4, num_theta_dot - 1, endpoint=False)

    self.dimension = num_x * num_x_dot * num_theta * num_theta_dot

  def __str__(self):
    return "TabularFeatures(num_x={}, num_x_dot={}, " \
                            "num_theta={}, num_theta_dot={})" \
            .format(self.num_x, self.num_x_dot,
                    self.num_theta, self.num_theta_dot)

  def _get_single_ind(self, var, var_bin):
    if len(var_bin) == 0:
      return 0
    else:
      return int(np.digitize(var, var_bin))

  def _get_state_num(self, x_ind, x_dot_ind, theta_ind, theta_dot_ind):
    state_num = \
      (x_ind + x_dot_ind * self.num_x
       + theta_ind * (self.num_x * self.num_x_dot)
       + theta_dot_ind * (self.num_x * self.num_x_dot * self.num_theta_dot))
    return int(state_num)

  def get_feature(self, obs):
    """We get the index using the linear space"""
    x, x_dot, theta, theta_dot = obs
    x_ind = self._get_single_ind(x, self.x_bins)
    x_dot_ind = self._get_single_ind(x_dot, self.x_dot_bins)
    theta_ind = self._get_single_ind(theta, self.theta_bins)
    theta_dot_ind = self._get_single_ind(theta_dot, self.theta_dot_bins)

    state_num = self._get_state_num(x_ind, x_dot_ind, theta_ind, theta_dot_ind)
    return state_num