import math
import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
import transformers
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer)

from jenga.models.modeling_llama import LlamaForCausalLM
from jenga.trainer.memory_profile import Trainer
from jenga.utils.config_utils import get_llama_qk
from jenga.utils.others import (seed_everything,
                                smart_tokenizer_and_embedding_resize)

BEGIN_TOKEN, END_TOKEN = '<<BEGIN>>', '<<END>>'
DEFAULT_PAD_TOKEN = "[PAD]"   #默认填充
DEFAULT_EOS_TOKEN = "</s>"    #句子结束
DEFAULT_BOS_TOKEN = "<s>"       #句子开始
DEFAULT_UNK_TOKEN = "<unk>"     #未知
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/home/pairshoe/cxy/models/llama3-1B")
    pack_loss: bool = field(default=False)
    predictor_path : str = field(default="experiment/train_distribute/output/llama3-8B_8192/checkpoint-20/")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    flash_attention: bool = field(default=False)
    pool_size: int = field(default=64)
    thresh: float = field(default=0.05)
    model_max_length: int = field(default=16384)


def set_RoPE(config,model_max_length):
    # Set RoPE scaling factor
    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}

    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    
    return config
 

def tokenize_fn(tokenizer, example):
    context_length = tokenizer.model_max_length
    outputs = tokenizer(
        tokenizer.eos_token.join(example["text"]),
        truncation=False,
        return_tensors="pt",
        pad_to_multiple_of=context_length,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}

def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    seed_everything(42)
    
    config = get_llama_qk(model_name=model_args.model_name_or_path,
                              flash_attention=training_args.flash_attention,
                              pool_size=training_args.pool_size,
                              thresh=training_args.thresh)
    
    config = set_RoPE(config, training_args.model_max_length)
    
    pruned_cfg = torch.load(os.path.join(model_args.predictor_path, "pruned_config.pth"))
    layers_cfg = pruned_cfg["layers"]
    config.predictor_layers = layers_cfg

    if training_args.bf16:
        model_param_type = torch.bfloat16
    else:
        model_param_type = torch.float32
        
    model = LlamaForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                        torch_dtype=model_param_type,
                                        config=config)
    attn_state_dict = torch.load(os.path.join(model_args.predictor_path,"predictor.pth"), map_location='cpu')
    for k,v in attn_state_dict.items():
        if k in model.state_dict():
            # print('load',k)
            model.state_dict()[k].copy_(v)
        else:
            print('skip',k)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                model_max_length=training_args.model_max_length,
                                                padding_size="right",
                                                use_fast=True)
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    if tokenizer.cls_token is None:
        special_tokens_dict['cls_token'] = BEGIN_TOKEN
    if tokenizer.sep_token is None:
        special_tokens_dict['sep_token'] = END_TOKEN
    special_tokens_dict['additional_special_tokens'] = ["[INST]", "[/INST]"]
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    
    dataset = load_dataset("./dataset/RedPajama-Data-1T-Sample", cache_dir=training_args.cache_dir,trust_remote_code=True)
    train_dataset = dataset['train']
    small_train_dataset = train_dataset.select(range(1000))

    # 创建新的 DatasetDict
    small_dataset = DatasetDict({
        'train': small_train_dataset
    })
    small_dataset = small_dataset.map(partial(tokenize_fn,tokenizer),batched=True, num_proc=4, remove_columns=["text", "meta"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    # [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in ["embed","norm"]])]
    # model.config.use_cache = False         # required for gradient checkpointing
    # model.enable_input_require_grads()     # required for gradient checkpointing
    # model.gradient_checkpointing_enable()  # enable gradient checkpointing
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=small_dataset["train"],
        eval_dataset=None,
        data_collator=data_collator)

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model()

    trainer.save_state()

if __name__ == "__main__":
    train()