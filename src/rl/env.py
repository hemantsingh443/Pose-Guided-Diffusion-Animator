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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        
        # Observation: Current angles (10) + Phase Signal (sin, cos) = 12
        self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(12,), dtype=np.float32)
        
        self.state = None
        self.phase = 0.0
        self.steps = 0
        self.max_steps = 200
        
        # Define Keyframes for Jumping Jack
        self.pose_stand = {
            'torso': 0, 'head': 0,
            'l_shoulder': -80, 'l_elbow': 0,
            'r_shoulder': -80, 'r_elbow': 0,
            'l_hip': -10, 'l_knee': 0,
            'r_hip': -10, 'r_knee': 0
        }
        
        self.pose_jack = {
            'torso': 0, 'head': 0,
            'l_shoulder': -150, 'l_elbow': 0, # Arms up
            'r_shoulder': -150, 'r_elbow': 0,
            'l_hip': -45, 'l_knee': 0,      # Legs out
            'r_hip': -45, 'r_knee': 0
        }
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.skeleton = StickFigure() 
        self.state = self.skeleton.get_random_pose()
        # Start near standing pose to help exploration
        for k, v in self.pose_stand.items():
            self.state[k] = v + np.random.randn() * 5
            
        self.phase = 0.0
        self.steps = 0
        return self._get_obs(), {}
        
    def _get_obs(self):
        keys = ['torso', 'head', 'l_shoulder', 'l_elbow', 'r_shoulder', 'r_elbow', 'l_hip', 'l_knee', 'r_hip', 'r_knee']
        angles = np.array([self.state[k] for k in keys], dtype=np.float32) / 180.0
        
        # Add phase signal
        phase_obs = np.array([np.sin(self.phase), np.cos(self.phase)], dtype=np.float32)
        
        return np.concatenate([angles, phase_obs])
        
    def step(self, action):
        keys = ['torso', 'head', 'l_shoulder', 'l_elbow', 'r_shoulder', 'r_elbow', 'l_hip', 'l_knee', 'r_hip', 'r_knee']
        
        # Apply action
        scale = 10.0 
        for i, k in enumerate(keys):
            self.state[k] += action[i] * scale
            # Clip to constraints
            min_a, max_a = self.skeleton.constraints[k]
            self.state[k] = np.clip(self.state[k], min_a, max_a)
            
        self.steps += 1
        self.phase += 0.2 # Advance phase
        
        # Calculate Target Pose based on Phase
        # Interpolate between Stand (-1) and Jack (+1)
        # sin(phase) goes -1 to 1
        alpha = (np.sin(self.phase) + 1) / 2.0 # 0 to 1
        
        total_error = 0.0
        for k in keys:
            target = self.pose_stand[k] * (1 - alpha) + self.pose_jack[k] * alpha
            total_error += abs(self.state[k] - target)
            
        # Normalize error (avg error per joint in degrees)
        avg_error = total_error / 10.0
        
        # Reward: exponential decay of error
        reward = np.exp(-avg_error / 30.0) 
        
        # Smoothness penalty
        reward -= np.sum(np.square(action)) * 0.01
        
        terminated = False
        truncated = self.steps >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}
        
    def render(self):
        return None
