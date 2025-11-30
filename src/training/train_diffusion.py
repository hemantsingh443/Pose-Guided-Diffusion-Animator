import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from ..diffusion.unet import UNet
from ..diffusion.model import Diffusion
import os

def train_diffusion_v2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 16
    lr = 3e-4
    epochs = 50
    
    # Load Data
    data_path = "data/stick_figure_dataset.pt"
    if not os.path.exists(data_path):
        print("Dataset not found! Run src/common/dataset.py first.")
        return

    data = torch.load(data_path)
    images = data['images']      # [N, 1, 64, 64]
    pose_maps = data['pose_maps'] # [N, 1, 64, 64]
    
    # Normalize images to [-1, 1]
    images = images * 2 - 1
    pose_maps = pose_maps * 2 - 1
    
    dataset = TensorDataset(images, pose_maps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model
    # c_in=2 (1 noisy image + 1 pose map)
    model = UNet(c_in=2, c_out=1, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=128, device=device)
    
    if not os.path.exists("models"):
        os.makedirs("models")

    print("Starting training with Pose Map conditioning...")
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for i, (images, pose_maps) in enumerate(pbar):
            images = images.to(device)
            pose_maps = pose_maps.to(device)
            
            t = diffusion.sample_timesteps(images.shape[0])
            x_t, noise = diffusion.noise_images(images, t)
            
            # Concatenate Pose Map to Noisy Input
            # x_input: [B, 2, 64, 64]
            x_input = torch.cat([x_t, pose_maps], dim=1)
            
            predicted_noise = model(x_input, t)
            
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(MSE=loss.item())
            
        # Save model
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), "models/ddpm_posemap.pt")
            print(f"Epoch {epoch+1}/{epochs} finished. Model saved.")

if __name__ == "__main__":
    train_diffusion_v2()
