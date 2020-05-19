from builtins import object
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

try:
  import roboschool
except:
  pass
import gym
import numpy as np
import pybullet_envs
from config import config
from gym import spaces
import copy

MAX_FRAMES = config["env"]["max_frames"]

gym.logger.level=40

def get_env(env_name, *args, **kwargs):
  # MAPPING = {
  #   "CartPole-v0": CartPoleWrapper,
  # }
  # if env_name in MAPPING: return MAPPING[env_name](env_name, *args, **kwargs)
  # else: return NoTimeLimitMujocoWrapper(env_name, *args, **kwargs)
  return RealerWalkerWrapper(env_name,*args, **kwargs)

class GymWrapper(object):
  """
  Generic wrapper for OpenAI gym environments.
  """
  def __init__(self, env_name):
    self.internal_env = gym.make(env_name)
    self.observation_space = self.internal_env.observation_space
    self.action_space = self.internal_env.action_space
    self.custom_init()

  def custom_init(self):
    pass

  def reset(self):
    self.clock = 0
    return self.preprocess_obs(self.internal_env.reset())

  # returns normalized actions
  def sample(self):
    return self.action_space.sample()

  # this is used for converting continuous approximations back to the original domain
  def normalize_actions(self, actions):
    return actions

  # puts actions into a form where they can be predicted. by default, called after sample()
  def unnormalize_actions(self, actions):
    return actions

  def preprocess_obs(self, obs):
    # return np.append(obs, [self.clock/float(MAX_FRAMES)])
    return obs

  def step(self, normalized_action):
    out = self.internal_env.step(normalized_action)
    self.clock += 1
    obs, reward, done = self.preprocess_obs(out[0]), out[1], float(out[2])
    reset = done == 1. or self.clock == MAX_FRAMES
    return obs, reward, done, reset

  def render_rollout(self, states):
    ## states is numpy array of size [timesteps, state]
    self.internal_env.reset()
    for state in states:
      self.internal_env.env.state = state
      self.internal_env.render()

class CartPoleWrapper(GymWrapper):
  """
  Wrap CartPole.
  """
  def sample(self):
    return np.array([np.random.uniform(0., 1.)])

  def normalize_actions(self, action):
    return 1 if action[0] >= 0 else 0

  def unnormalize_actions(self, action):
    return 2. * action - 1.

class NoTimeLimitMujocoWrapper(GymWrapper):
  """
  Wrap Mujoco-style environments, removing the termination condition after time.
  This is needed to keep it Markovian.
  """
  def __init__(self, env_name):
    self.internal_env = gym.make(env_name).env
    self.observation_space = self.internal_env.observation_space
    self.action_space = self.internal_env.action_space
    self.custom_init()




class RealerWalkerWrapper(gym.Env):
    def __init__(self, env_name, ep_len=100, rew='walk'):
        self.internal_env = gym.make(env_name)

        # obs_scale = 5
        obs_scale = 1
        act_scale = 1

        self.action_space = self.internal_env.action_space
        
        
        # self.action_space.high *= act_scale
        # self.action_space.low *= act_scale

        self.front = 3 # first 3 need to be skipped as they are angle to target, 2nd 3 (totaling 6) account for velocity
        self.back = 4 # cutting out the feet contacts
        obs_ones = obs_scale*np.ones(shape=(self.internal_env.observation_space.shape[0]-self.front-self.back,))
        self.observation_space = spaces.Box(high=self.internal_env.observation_space.high[self.front:-self.back],
                                            low=self.internal_env.observation_space.low[self.front:-self.back])
        # self.observation_space = self.internal_env.observation_space.high
        # State Summary (dim=25):
        # state[0] = vx
        # state[1] = vy
        # state[2] = vz
        # state[3] = roll
        # state[4] = pitch
        # state[5 to -4] = Joint relative positions
        #    even elements [0::2] position, scaled to -1..+1 between limits
        #    odd elements  [1::2] angular speed, scaled to show -1..+1
        # state[-4 to -1] = feet contacts
        self.timestep = 0
        self.max_time = ep_len
        self.rew = rew
        self.custom_init()


    def reset(self):
        obs = self.internal_env.reset()
        # Could clip to +/- 5 since thats what they do in pybullet_envs robot_locomotors.py
        # obs =  np.clip(obs[self.front:-self.back], -5, +5)
        obs = obs[self.front:-self.back]
        self.timestep = 0
        self.ep_rew = 0
        self.clock = 0
        return obs


    def step(self, action):
        new_obs, _, done, info = self.internal_env.step(action)
        # r = (new_obs[3]/0.3)/60 # x velocity
        # r = new_obs[4] # y velocity
        if self.rew == 'walk':
            r = new_obs[3]
        elif self.rew == 'jump':
            r = max(new_obs[5], 0) # Only the positive Z velocity (i.e. don't penalize for falling after the jump)
        else:
            r = new_obs[3]

        # Could clip to +/- 5 since thats what they do in pybullet_envs robot_locomotors.py
        # obs =  np.clip(obs[self.front:-self.back], -5, +5)
        new_obs = new_obs[self.front:-self.back]
        self.timestep += 1
        done = self.timestep >= self.max_time
        self.ep_rew += r


        self.clock += 1
        

        #doing no preprocessing
        # obs, reward, done = self.preprocess_obs(out[0]), out[1], float(out[2])
    
  
        reset = done == 1. or self.clock == MAX_FRAMES

        # info = {}
        # if done:
        #     info['episode'] = {}
        #     info['episode']['r'] = self.ep_rew
        #     info['episode']['l'] = self.timestep
        return new_obs, r, done, reset

    def reset_raw(self):
        obs = self.internal_env.reset()
        return obs
    def step_raw(self, action):
        new_obs, r, done, info = self.internal_env.step(action)
        return new_obs, r, done, info
    def render(self, mode='human'):
        return self.internal_env.render(mode)
  
    def render_rollout(self, states):
      ## states is numpy array of size [timesteps, state]
      self.internal_env.reset()
      for state in states:
        self.internal_env.env.state = state
        self.internal_env.render()


    def sample(self):
      return self.action_space.sample()

    # this is used for converting continuous approximations back to the original domain
    def normalize_actions(self, actions):
      return actions

    # puts actions into a form where they can be predicted. by default, called after sample()
    def unnormalize_actions(self, actions):
      return actions

    def preprocess_obs(self, obs):
      # return np.append(obs, [self.clock/float(MAX_FRAMES)])
      return obs
    def custom_init(self):
      pass


    # def seed(self, s):
    #     return self.internal_env.seed(s)