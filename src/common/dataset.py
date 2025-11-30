import os
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from .skeleton import StickFigure
from .renderer import StickFigureRenderer

def generate_dataset(num_samples=10000, image_size=(128, 128), output_dir="data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    skeleton = StickFigure()
    # Target renderer
    renderer_target = StickFigureRenderer(image_size=image_size, line_width=2)
    # Condition renderer
    renderer_cond = StickFigureRenderer(image_size=image_size, line_width=2)
    
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(), 
    ])
    
    images = []
    pose_maps = []
    
    print(f"Generating {num_samples} samples (Target + Pose Map)...")
    for _ in tqdm(range(num_samples)):
        pose = skeleton.get_random_pose()
        joints = skeleton.forward_kinematics(pose)
        
        # Render Target
        img = renderer_target.render(joints, skeleton_config=skeleton)
        img_tensor = transform(img)
        
        # Render Pose Map (Condition)
        pose_img = renderer_cond.render(joints, skeleton_config=skeleton)
        pose_tensor = transform(pose_img)
        
        images.append(img_tensor)
        pose_maps.append(pose_tensor)
        
    images = torch.stack(images)
    pose_maps = torch.stack(pose_maps)
    
    output_path = os.path.join(output_dir, "stick_figure_dataset.pt")
    torch.save({'images': images, 'pose_maps': pose_maps}, output_path)
    print(f"Dataset saved to {output_path}")
    print(f"Images shape: {images.shape}")
    print(f"Pose Maps shape: {pose_maps.shape}")
    
    # Save sample
    sample_img = transforms.ToPILImage()(images[0])
    sample_pose = transforms.ToPILImage()(pose_maps[0])
    sample_img.save("sample_target.png")
    sample_pose.save("sample_pose.png")
    print("Saved sample_target.png and sample_pose.png")

if __name__ == "__main__":
    generate_dataset()
