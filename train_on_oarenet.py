import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.radanet import RADANet
from dataset import RoadDataset
from utils.losses import CombinedLoss
import argparse

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training RADANet on OARENet Data ({device}) ---")
    
    # Init Model
    model = RADANet(num_classes=1, pretrained=True).to(device)
    
    # OARENet Data Paths
    img_dir = args.img_dir
    mask_dir = args.mask_dir
    
    print(f"Loading data from:\nImages: {img_dir}\nLabels: {mask_dir}")
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print("Error: OARENet data directories not found.")
        return

    # Dataset & Dataloader
    dataset = RoadDataset(img_dir=img_dir, mask_dir=mask_dir, img_size=(args.img_size, args.img_size), train=True)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Loss, Optimizer
    criterion = CombinedLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training Loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i+1) % max(1, len(train_loader)//5) == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss/len(train_loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "radanet_oarenet_best.pth")
            print("Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RADANet on OARENet Dataset")
    parser.add_argument("--img_dir", type=str, default="OARENet-main/OARENet-main/demo_data/img", help="OARENet image directory")
    parser.add_argument("--mask_dir", type=str, default="OARENet-main/OARENet-main/demo_data/label", help="OARENet mask directory")
    parser.add_argument("--img_size", type=int, default=256, help="Input size")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    
    args = parser.parse_args()
    train(args)
