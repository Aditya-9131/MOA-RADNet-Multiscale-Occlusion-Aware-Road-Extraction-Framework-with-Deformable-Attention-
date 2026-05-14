import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.radanet import RADANet
from dataset import RoadDataset
import numpy as np
from tqdm import tqdm
import argparse

def calculate_metrics(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    
    return precision, recall, f1, iou

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Evaluating RADANet on {device} ---")
    
    model = RADANet(num_classes=1, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.eval()
    
    dataset = RoadDataset(img_dir=args.img_dir, mask_dir=args.mask_dir, img_size=(args.img_size, args.img_size), train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    metrics = {'precision': [], 'recall': [], 'f1': [], 'iou': []}
    
    with torch.no_grad():
        for images, masks in tqdm(loader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            p, r, f, i = calculate_metrics(outputs, masks)
            
            metrics['precision'].append(p)
            metrics['recall'].append(r)
            metrics['f1'].append(f)
            metrics['iou'].append(i)
            
    print("\n--- Final Metrics ---")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {np.mean(v):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RADANet Model")
    parser.add_argument("--img_dir", type=str, default="OARENet-main/OARENet-main/demo_data/img", help="Image directory")
    parser.add_argument("--mask_dir", type=str, default="OARENet-main/OARENet-main/demo_data/label", help="Mask directory")
    parser.add_argument("--model_path", type=str, default="radanet_oarenet_best.pth", help="Path to checkpoint")
    parser.add_argument("--img_size", type=int, default=256, help="Input size")
    
    args = parser.parse_args()
    evaluate(args)
