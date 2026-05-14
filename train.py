import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model import RADANet
from dataset import DeepGlobeDataset
from utils.losses import ImprovedRADALoss
from utils.plots import plot_training_results, plot_sample_results
import argparse
from tqdm import tqdm

def calculate_metrics(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    
    return precision, recall, f1, iou

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training improved RADANet on {device} ---")
    
    # 1. Init Model
    model = RADANet(num_classes=1).to(device)
    
    # 2. Datasets & Loaders
    train_ds = DeepGlobeDataset(args.img_dir, args.mask_dir, img_size=args.img_size, mode='train')
    val_ds = DeepGlobeDataset(args.img_dir, args.mask_dir, img_size=args.img_size, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 3. Loss & Optimizer
    criterion = ImprovedRADALoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Trace history
    history = {'loss': [], 'val_iou': []}
    best_iou = 0.0
    
    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        # Validation
        model.eval()
        val_ious = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                _, _, _, iou = calculate_metrics(outputs, masks)
                val_ious.append(iou)
        
        avg_iou = sum(val_ious) / len(val_ious)
        history['val_iou'].append(avg_iou)
        scheduler.step()
        
        print(f"Epoch {epoch+1} Summary: Avg Loss: {avg_loss:.4f}, Val IoU: {avg_iou:.4f}")
        
        # Save Best Model
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), "radanet_best_deepglobe.pth")
            print(f"  --> Saved Best Model (IoU: {best_iou:.4f})")
            
            # Save a sample prediction
            sample_img, sample_mask = val_ds[0]
            with torch.no_grad():
                pred = model(sample_img.unsqueeze(0).to(device))
            plot_sample_results(sample_img, sample_mask, pred.squeeze(0), "latest_val_sample.png")

        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
            plot_training_results(history['loss'], history['val_iou'])

    print("Training Complete!")
    plot_training_results(history['loss'], history['val_iou'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    train(args)
