import functools
import inspect
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from transformers import Trainer
from jenga.models.predictor import PrunableAttnPredictor


def extract_pruned_config(model: nn.Module):
    """
    遍历模型中的 PrunableAttnPredictor1，收集它们当前 out_features。
    假设我们只需要存 predictor 的 outdim 信息，用于下次构造同样剪枝形状的 Predictor。
    当然，你也可以存更多自定义字段。
    
    返回一个 dict，比如:
    {
      "layers": [
         {
           "q1_outdim": X,
           "q2_outdim": Y,
           "k1_outdim": Z,
           "k2_outdim": W
         },
         ...
      ]
    }
    """
    cfg = {"layers": []}
    for layer in model.model.layers:
        predictor = layer.self_attn.predictor
        if isinstance(predictor, PrunableAttnPredictor):
            layer_cfg = predictor.get_current_outdims()  # dict
            cfg["layers"].append(layer_cfg)
        else:
            cfg["layers"].append(None)
    return cfg


class PredictorTrainer(Trainer):
    def __init__(self, orig_weight_training=False, gate_loss_scale=1.0, fix_mask_predictor=False, use_mse_loss=False,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        if use_mse_loss:
            self.loss_fct = torch.nn.MSELoss()
        else:
            self.loss_fct = torch.nn.SmoothL1Loss()
        self.gate_loss_scale = gate_loss_scale
        self.orig_weight_training = orig_weight_training
        self.fix_mask_predictor = fix_mask_predictor


    def compute_loss(self, model, inputs, return_outputs=False):
        step = self.state.global_step
        outputs = model(**inputs)

        predict_mask = outputs.get("predict_mask")
        pooling_gt = outputs.get("pooling_gt")
        original_loss = outputs.get("loss")

        if not return_outputs:
            del outputs

        mask_loss = 0
        if not self.fix_mask_predictor:
            for pm, gt in zip(predict_mask, pooling_gt):
                mask_loss += self.loss_fct(pm, gt)

          
        del predict_mask
        del pooling_gt
        
        
        if self.orig_weight_training:
            tok_loss = original_loss + self.gate_loss_scale * mask_loss
        else:
            tok_loss = self.gate_loss_scale * mask_loss
        
        return (tok_loss, outputs) if return_outputs else tok_loss
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        pruned_cfg = extract_pruned_config(self.model)
        params_to_save = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
        torch.save(params_to_save, os.path.join(output_dir, "predictor.pth"))
        torch.save(pruned_cfg, os.path.join(output_dir, "pruned_config.pth"))
    
class DynamicPruningPredictorTrainer(PredictorTrainer):
    """
    在 PredictorTrainer 基础上，增加每隔 prune_interval 步，对模型中
    每个 PrunableAttnPredictor1 执行一次 prune_neurons() 的逻辑.
    """
    def __init__(self, prune_interval=100, zero_ratio_threshold=0.8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prune_interval = prune_interval
        self.zero_ratio_threshold = zero_ratio_threshold

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_tuple = super().compute_loss(model, inputs, return_outputs=True)
        step = self.state.global_step

        # 每隔 self.prune_interval 步，对 predictor 执行一次剪枝
        if step > 0 and (step % self.prune_interval == 0) and step < 620:
            times = step // self.prune_interval
            thresh  = self.zero_ratio_threshold - (times - 1)*0.05
            for module_name, module in model.named_modules():
                if isinstance(module, PrunableAttnPredictor):
                    module.prune_neurons(
                        step_count_threshold=self.prune_interval,
                        zero_ratio_threshold=thresh
                    )
        
        if return_outputs:
            return loss_tuple
        else:
            return loss_tuple[0]