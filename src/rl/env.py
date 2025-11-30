import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ..common.skeleton import StickFigure

class StickFigureEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.skeleton = StickFigure()
        self.render_mode = render_mode
        
        # Action: Delta for each joint angle (10 joints)
        # We scale actions to be small updates per step for smoothness
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        
        # Observation: Current angles (10) + Velocities (10) could be better, but let's stick to angles for now
        # Normalized angles roughly [-1, 1]
        self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(10,), dtype=np.float32)
        
        self.state = None
        self.steps = 0
        self.max_steps = 200 # Longer episodes
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.skeleton = StickFigure() 
        self.state = self.skeleton.get_random_pose()
        self.steps = 0
        return self._get_obs(), {}
        
    def _get_obs(self):
        keys = ['torso', 'head', 'l_shoulder', 'l_elbow', 'r_shoulder', 'r_elbow', 'l_hip', 'l_knee', 'r_hip', 'r_knee']
        obs = np.array([self.state[k] for k in keys], dtype=np.float32)
        return obs / 180.0
        
    def step(self, action):
        keys = ['torso', 'head', 'l_shoulder', 'l_elbow', 'r_shoulder', 'r_elbow', 'l_hip', 'l_knee', 'r_hip', 'r_knee']
        
        # Apply action (scaled)
        scale = 10.0 # Increased for more motion
        
        for i, k in enumerate(keys):
            change = action[i] * scale
            self.state[k] += change
            
            # Clip to constraints
            min_a, max_a = self.skeleton.constraints[k]
            self.state[k] = np.clip(self.state[k], min_a, max_a)
            
        self.steps += 1
        
        # Reward: Dynamic Waving (Track a Sine Wave with Left Arm)
        # Target oscillates between -130 and -50 degrees (Arm waving up and down)
        target_angle = -90 + np.sin(self.steps * 0.2) * 40
        
        # Normalize error
        angle_error = abs(self.state['l_shoulder'] - target_angle) / 180.0
        
        reward = 1.0 - angle_error
        
        # Smoothness penalty
        reward -= np.sum(np.square(action)) * 0.01
        
        terminated = False
        truncated = self.steps >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}
        
    def render(self):
        # use the diffusion model later
        return None
