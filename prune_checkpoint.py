import torch
import os

def prune(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Pruning {input_path} (~{os.path.getsize(input_path)/1e9:.2f} GB)...")
    print("This may take a moment and requires available RAM.")
    
    try:
        # map_location='cpu' ensures we don't try to use GPU
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            # If it's a dict, we want 'model_state_dict'
            if 'model_state_dict' in checkpoint:
                clean_state_dict = checkpoint['model_state_dict']
                print("✓ Found and extracted 'model_state_dict'")
            else:
                # If it's a dict but no model_state_dict, maybe the dict IS the state dict?
                # Check if keys look like model weights (e.g., 'enc0.0.weight')
                keys = list(checkpoint.keys())
                if any('weight' in k or 'bias' in k for k in keys):
                    clean_state_dict = checkpoint
                    print("✓ Dictionary appears to be the state dict itself.")
                else:
                    print("! Warning: Dictionary found but no obvious weights. Saving full dict.")
                    clean_state_dict = checkpoint
        else:
            # If it's not a dict, it's likely the model itself or the raw state dict
            clean_state_dict = checkpoint
            print("✓ Loaded raw state dict / model object.")
            
        torch.save(clean_state_dict, output_path)
        print(f"✅ Successfully saved pruned weights to {output_path} (~{os.path.getsize(output_path)/1e9:.2f} GB)")
        
    except Exception as e:
        print(f"❌ Error during pruning: {e}")
        print("\nTry the following:")
        print("1. Restart your computer to clear standby RAM.")
        print("2. Close Chrome and other memory-heavy applications.")
        print("3. Use the 'radanet_latest.pth' file instead, which is already pruned.")

if __name__ == "__main__":
    # Prioritize pruning the 3.9GB file
    prune("best_model.pth", "best_model_pruned.pth")
    
    # Also prune the 1.3GB file just in case
    if os.path.exists("best_radanet_cpu.pth") and not os.path.exists("best_radanet_cpu_pruned.pth"):
        prune("best_radanet_cpu.pth", "best_radanet_cpu_pruned.pth")
