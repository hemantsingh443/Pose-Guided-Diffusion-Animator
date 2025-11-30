import torch
import torchvision
from torchvision import transforms
from ..diffusion.unet import UNet
from ..diffusion.model import Diffusion
from ..rl.env import StickFigureEnv
from ..common.renderer import StickFigureRenderer
from stable_baselines3 import PPO
import os
import numpy as np
from PIL import Image

def animate_v2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load RL Agent
    env = StickFigureEnv()
    rl_path = "models/ppo_stickfigure.zip"
    
    if os.path.exists(rl_path):
        model = PPO.load(rl_path)
        print("Loaded PPO agent.")
    else:
        print("RL agent not found! Run src/training/train_rl_sb3.py first.")
        return
        
    # 2. Generate Pose Sequence & Render Pose Maps
    print("Generating pose sequence...")
    obs, _ = env.reset()
    
    renderer = StickFigureRenderer(image_size=(128, 128), line_width=2)
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(), 
    ])
    
    pose_maps = []
    
    T = 64 
    for _ in range(T):
        # Reconstruct pose from obs (obs is normalized)
        
        # Predict action
        action, _states = model.predict(obs, deterministic=True)
        
        # Step env
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render current state to Pose Map
        joints = env.skeleton.forward_kinematics(env.state)
        pose_img = renderer.render(joints, skeleton_config=env.skeleton)
        pose_tensor = transform(pose_img) # [1, 128, 128]
        pose_maps.append(pose_tensor)

        if terminated or truncated:
            obs, _ = env.reset()

    pose_maps = torch.stack(pose_maps).to(device) # [T, 1, 128, 128]
    pose_maps = pose_maps * 2 - 1 # Normalize to [-1, 1]
    
    # 3. Load Diffusion Model
    diff_model = UNet(c_in=2, c_out=1, device=device).to(device)
    diff_path = "models/ddpm_posemap.pt"
    if os.path.exists(diff_path):
        diff_model.load_state_dict(torch.load(diff_path))
        print("Loaded Diffusion model.")
    else:
        print("Diffusion model not found! Exiting.")
        return

    diffusion = Diffusion(img_size=128, device=device)
    
    # 4. Render Frames
    print(f"Rendering {T} frames with Pose Map conditioning...")
    
    # Custom sampling loop to handle concatenation
    diff_model.eval()
    with torch.no_grad():
        x = torch.randn((T, 1, 128, 128)).to(device)
        for i in reversed(range(1, diffusion.noise_steps)):
            t = (torch.ones(T) * i).long().to(device)
            
            # Concatenate Pose Map
            x_input = torch.cat([x, pose_maps], dim=1)
            
            predicted_noise = diff_model(x_input, t)
            
            alpha = diffusion.alpha[t][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            beta = diffusion.beta[t][:, None, None, None]
            
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
    diff_model.train()
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    
    # 5. Save Animation
    imgs = [transforms.ToPILImage()(img) for img in x]
    imgs[0].save("final_posemap_animation.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)
    print("Saved final_posemap_animation.gif")

if __name__ == "__main__":
    animate_v2()
