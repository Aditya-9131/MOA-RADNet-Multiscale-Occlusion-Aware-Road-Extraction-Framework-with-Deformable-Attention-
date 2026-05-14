import os
import numpy as np
from PIL import Image, ImageDraw
import random

def generate_dummy_road(size=(512, 512)):
    # Create black mask
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    
    # Create image with random background (e.g., green for landscape)
    bg_color = (random.randint(30, 100), random.randint(100, 200), random.randint(30, 100))
    img = Image.new('RGB', size, bg_color)
    draw_img = ImageDraw.Draw(img)
    
    # Draw some random lines as "roads"
    num_roads = random.randint(2, 5)
    for _ in range(num_roads):
        points = []
        curr_x = random.randint(0, size[0])
        curr_y = random.randint(0, size[1])
        points.append((curr_x, curr_y))
        
        for _ in range(4):
            curr_x = int(np.clip(curr_x + random.randint(-150, 150), 0, size[0]))
            curr_y = int(np.clip(curr_y + random.randint(-150, 150), 0, size[1]))
            points.append((curr_x, curr_y))
        
        width = random.randint(6, 12)
        # Draw on mask
        draw.line(points, fill=255, width=width, joint='curve')
        # Draw on image (grayish road)
        draw_img.line(points, fill=(180, 180, 180), width=width, joint='curve')
        
    return img, mask

def create_dataset(base_dir, num_samples=10):
    img_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'masks')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    print(f"Generating {num_samples} samples into {base_dir}...")
    for i in range(num_samples):
        img, mask = generate_dummy_road()
        img.save(os.path.join(img_dir, f"sample_{i}.png"))
        mask.save(os.path.join(mask_dir, f"sample_{i}.png"))

if __name__ == "__main__":
    # Ensure directories exist relative to project root
    create_dataset('data/train', num_samples=20)
    create_dataset('data/val', num_samples=10)
    print("Dummy dataset creation complete.")
