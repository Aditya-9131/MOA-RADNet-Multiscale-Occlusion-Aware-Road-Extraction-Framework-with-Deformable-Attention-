import torch
import torch.nn.functional as F
from models.radanet import RADANet
from PIL import Image
import torchvision.transforms as T
import numpy as np
import argparse
import os

def predict(img_path, model_path, output_path, img_size=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    
    # Init and load model
    model = RADANet(num_classes=1, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    
    # Preprocess
    img = Image.open(img_path).convert("RGB")
    orig_size = img.size
    
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = transform(img).unsqueeze(0).to(device)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        out = model(x)
        out = torch.sigmoid(out)
        out = F.interpolate(out, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=True)
        out_mask = (out > 0.5).float()
    
    # Save output
    out_img = out_mask.squeeze().cpu().numpy() * 255
    res = Image.fromarray(out_img.astype(np.uint8))
    res.save(output_path)
    print(f"Saved prediction to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RADANet Inference Script")
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="prediction.png", help="Path to save output mask")
    parser.add_argument("--img_size", type=int, default=512, help="Input size for model")
    
    args = parser.parse_args()
    
    if os.path.exists(args.img) and os.path.exists(args.model):
        predict(args.img, args.model, args.output, args.img_size)
    else:
        print("Error: Image or model file not found.")
