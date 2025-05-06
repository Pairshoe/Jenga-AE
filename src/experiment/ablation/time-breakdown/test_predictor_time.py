import argparse
import torch
import os
from jenga.models.predictor import PrunableAttnPredictorInfer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--predictor_path", type=str, default="checkpoints/predictor")
    args = parser.parse_args()
    
    pruned_cfg = torch.load(os.path.join(args.predictor_path, "pruned_config.pth"))
    layers_cfg = pruned_cfg["layers"]
    
    predictors = []

    for layer in range(len(layers_cfg)):
        layer_cfg = layers_cfg[layer]
        predictor = PrunableAttnPredictorInfer(dim=128,
                                               hidden_dim=512,
                                               q1_outdim=layer_cfg["q1_outdim"],
                                               q2_outdim=layer_cfg["q2_outdim"],
                                               k1_outdim=layer_cfg["k1_outdim"],
                                               k2_outdim=layer_cfg["k2_outdim"],)
        predictor = predictor.to(torch.bfloat16)
        predictors.append(predictor)
        
    times = []
    
    warmup_runs = 5  # Warm-up to stabilize CUDA performance
    for i in range(len(predictors)):
        predictor = predictors[i].to("cuda")
        predictor.eval()
        with torch.no_grad():
            input = torch.randn(1, args.seq_len, 4096).to(torch.bfloat16).to("cuda")

            # Warm-up runs
            for _ in range(warmup_runs):
                _ = predictor(input)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            
            output = predictor(input)
            
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)  # ms
            times.append(elapsed_time)
            # print(f"Layer {i} time: {elapsed_time} ms")
    
    print(f"seq_len: {args.seq_len} time: {sum(times)} ms")
        
    
    