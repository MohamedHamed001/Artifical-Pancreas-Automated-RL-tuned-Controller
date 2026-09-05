import gymnasium as gym
import numpy as np

class NormalizeObservation(gym.ObservationWrapper):
    """
    Standardizes observations using running mean and standard deviation.
    """
    def __init__(self, env):
        super().__init__(env)
        self.mean = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.std = np.ones(env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return (observation - self.mean) / (self.std + 1e-8)

class ClipAction(gym.ActionWrapper):
    """
    Ensures actions stay within bounds.
    """
    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)
