import torch
import argparse 
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prunes non-inference weights from XTTSv2 models to save disk space')
    parser.add_argument('model_path', help='Model to prune.')
    args = parser.parse_args()

    checkpoint = torch.load(args.model_path, map_location = torch.device("cpu"))
    del checkpoint["optimizer"]
    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]
    save_path = Path(args.model_path).stem+'_pruned.pth'
    torch.save(checkpoint, save_path)
    print(f"Saved to {save_path}")