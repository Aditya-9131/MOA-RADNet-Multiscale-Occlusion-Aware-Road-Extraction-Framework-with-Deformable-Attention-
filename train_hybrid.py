import os
import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.hybrid_radanet import HybridRADANet
from utils.hybrid_losses import HybridLoss
from OARENet.dataset import RoadDataset
import argparse
from tqdm import tqdm

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training Hybrid-RADANet on {device} (D: Drive) ---")
    
    # 1. Model
    model = HybridRADANet(num_classes=1, pretrained=True).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {params/1e6:.2f}M")
    
    # 2. Data (Flexible path discovery)
    train_dir = os.path.join(args.dataroot, 'train')
    val_dir = os.path.join(args.dataroot, 'val')
    
    # Check for OARENet style or single-folder datasets
    if not os.path.exists(train_dir):
        train_dir = args.dataroot # Fallback to root
    if not os.path.exists(val_dir):
        val_dir = os.path.join(args.dataroot, 'valid')
    if not os.path.exists(val_dir):
        val_dir = args.dataroot # Fallback to root
    
    train_ds = RoadDataset(train_dir, img_size=args.img_size, mode='train')
    val_ds   = RoadDataset(val_dir,   img_size=args.img_size, mode='val')
    
    # CPU optimized workers
    num_workers = 0 if device.type == 'cpu' else 2
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    # 3. Loss & Optimizer
    criterion = HybridLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Modern AMP Setup
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == 'cuda'))
    
    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Explicit memory cleanup for CPU
            del outputs, loss
            if (i+1) % 5 == 0:
                gc.collect()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"hybrid_radanet_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated default to OARENet Demo dataset
    parser.add_argument("--dataroot", type=str, default="OARENet-main/OARENet-main/demo_data")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
