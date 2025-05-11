import random
from typing import Dict

import numpy as np
import torch
import transformers
import torch.nn as nn
hidden_statess = []
attn_maxpools = []


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer)) # 调整模型嵌入层的大小以匹配tokenizer的长度

    # 如果有新增的符号，更新这些新符号的嵌入使之与已有的嵌入平均值一致
    if num_new_tokens > 0:
        #已有权重
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        #增加新符号权重
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    return num_new_tokens

def load_state_dict_into_model_with_deepspeed_stage3(model_to_load, state_dict, start_prefix, assign_to_params_buffers=False):
    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    renamed_keys = {}
    renamed_gamma = {}
    renamed_beta = {}
    warning_msg = f"A pretrained model of type `{model_to_load.__class__.__name__}` "
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            # We add only the first key as an example
            new_key = key.replace("gamma", "weight")
            renamed_gamma[key] = new_key if not renamed_gamma else renamed_gamma
        if "beta" in key:
            # We add only the first key as an example
            new_key = key.replace("beta", "bias")
            renamed_beta[key] = new_key if not renamed_beta else renamed_beta
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    renamed_keys = {**renamed_gamma, **renamed_beta}
    if renamed_keys:
        warning_msg += "contains parameters that have been renamed internally (a few are listed below but more are present in the model):\n"
        for old_key, new_key in renamed_keys.items():
            warning_msg += f"* `{old_key}` -> `{new_key}`\n"
        warning_msg += "If you are using a model from the Hub, consider submitting a PR to adjust these weights and help future users."
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix="", assign_to_params_buffers=False):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        local_metadata["assign_to_params_buffers"] = assign_to_params_buffers

        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            import deepspeed

            # In sharded models, each shard has only part of the full state_dict, so only gather
            # parameters that are in the current state_dict.
            named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
            params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
            if len(params_to_gather) > 0:
                # because zero3 puts placeholders in model params, this context
                # manager gathers (unpartitions) the params of the current layer, then loads from
                # the state dict and then re-partitions them again
                with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".", assign_to_params_buffers)

    load(model_to_load, state_dict, prefix=start_prefix, assign_to_params_buffers=assign_to_params_buffers)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return error_msgs


