import argparse
import torch
import os
from jenga.models.predictor import PrunableAttnPredictorInfer, PrunableAttnPredictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor_path", type=str, default="logs/ablations/predictor/llama2_16384_predictor_red")
    args = parser.parse_args()
    
    pruned_cfg = torch.load(os.path.join(args.predictor_path, "pruned_config.pth"))
    layers_cfg = pruned_cfg["layers"]
    
    predictors = []
    total_params = 0

    for layer in range(len(layers_cfg)):
        layer_cfg = layers_cfg[layer]
        predictor = PrunableAttnPredictorInfer(dim=256,
                                               hidden_dim=64,
                                               q1_outdim=layer_cfg["q1_outdim"],
                                               q2_outdim=layer_cfg["q2_outdim"],
                                               k1_outdim=layer_cfg["k1_outdim"],
                                               k2_outdim=layer_cfg["k2_outdim"],)
        
        predictor_ori = PrunableAttnPredictor(dim=256, hidden_dim=64, n_head=32)
        
        params_ori = sum(p.numel() for p in predictor_ori.parameters())
        params = sum(p.numel() for p in predictor.parameters())
        print(f"Layer {layer}, Pruned params: {params}, original params: {params_ori}, Pct : {params / params_ori:.2f}")
        total_params += params
    
    total_memory = total_params * 2 / (1024 ** 2)  # Convert to MB
    
    print(f"Total memory: {total_memory:.2f} MB")
        