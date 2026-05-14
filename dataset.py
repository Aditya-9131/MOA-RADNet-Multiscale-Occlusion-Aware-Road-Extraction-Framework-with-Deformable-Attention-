import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import random

class DeepGlobeDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=512, mode='train'):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.mode = mode
        
        # Consistent list of images
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Split logic (70/15/15)
        # Note: In a real scenario, we would use a seed for reproducible splits
        n = len(self.images)
        indices = list(range(n))
        random.seed(42)
        random.shuffle(indices)
        
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        if mode == 'train':
            self.indices = indices[:train_end]
        elif mode == 'val':
            self.indices = indices[train_end:val_end]
        else: # test
            self.indices = indices[val_end:]

    def transform(self, image, mask):
        # Resize
        image = TF.resize(image, (self.img_size, self.img_size))
        mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.NEAREST)
        
        if self.mode == 'train':
            # Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Vertical Flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                
            # Random Rotation (0-360)
            angle = random.uniform(0, 360)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
            
            # Color Jitter (Image only)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
                image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
            
            # Gaussian Noise (Simulated by adding random tensor)
            if random.random() > 0.3:
                img_np = np.array(image).astype(np.float32)
                noise = np.random.normal(0, 0.05, img_np.shape) * 255
                img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(img_np)

        # Normalize and ToTensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        mask = TF.to_tensor(mask) # Scaled to [0, 1]
        
        return image, mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png")) # Handle common extensions
        
        # Ensure mask exists, skip if not or just assume consistency
        if not os.path.exists(mask_path):
            # Fallback if names don't match exactly
            mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Segmentation mask as Greyscale
        
        image, mask = self.transform(image, mask)
        
        return image, mask

if __name__ == "__main__":
    # Test loader
    img_dir = "data/train/images"
    mask_dir = "data/train/masks"
    if os.path.exists(img_dir):
        ds = DeepGlobeDataset(img_dir, mask_dir)
        img, mask = ds[0]
        print(f"Dataset items: {len(ds)}")
        print(f"Image tensor shape: {img.shape}")
        print(f"Mask tensor shape: {mask.shape}")
