import gymnasium as gym
from stable_baselines3 import PPO
from ..rl.env import StickFigureEnv
import os

def train_rl_sb3():
    # Create environment
    env = StickFigureEnv()
    
    # Initialize PPO Agent
    # MlpPolicy is standard for vector inputs
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10)
    
    print("Training RL Agent (PPO)...")
    # Train for 10k steps
    model.learn(total_timesteps=10000)
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    model.save("models/ppo_stickfigure")
    print("RL Agent saved to models/ppo_stickfigure.zip")

if __name__ == "__main__":
    train_rl_sb3()
