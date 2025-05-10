import argparse
import torch
import os
from jenga.models.predictor import PrunableAttnPredictorInfer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor_path", type=str, default="checkpoints/predictor")
    args = parser.parse_args()
    

    weight = torch.load(os.path.join(args.predictor_path, "predictor.pth"))

    # 计算显存占用（单位：MB）
    total_bytes = 0
    for k, v in weight.items():
        if isinstance(v, torch.Tensor):
            total_bytes += v.numel() * v.element_size()

    total_mb = total_bytes * (4+4+2) / (1024 ** 2)
    print(f"Total memory: {total_mb:.2f} MB")
    
   
        