import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from jenga.utils.LongAlign import LMDataset, LMPackDataset, LMSortDataset
from jenga.trainer.predictor_trainer import DynamicPruningPredictorTrainer
from jenga.models.modeling_llama_train_predictor import LlamaForCausalLM
from jenga.utils.config_utils import get_llama_qk
from jenga.utils.others import (seed_everything,
                               smart_tokenizer_and_embedding_resize)

BEGIN_TOKEN, END_TOKEN = '<<BEGIN>>', '<<END>>'
DEFAULT_PAD_TOKEN = "[PAD]"   #默认填充
DEFAULT_EOS_TOKEN = "</s>"    #句子结束
DEFAULT_BOS_TOKEN = "<s>"       #句子开始
DEFAULT_UNK_TOKEN = "<unk>"     #未知


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/home/pairshoe/cxy/models/llama3-1B")
    pack_loss: bool = field(default=False)

@dataclass
class DataArguments:
    train_file: str = field(default="data/llama/32k", metadata={"help": "Path to the training data."})
    validation_file: str = field(default=None, metadata={"help": "Path to the training data."})
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    prompt_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    response_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    batch_method: str = field(default="sort")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    flash_attention: bool = field(default=False)
    thresh: float = field(default=1/16)
    model_max_length: int = field(default=16384)

@dataclass
class DataCollatorForLMDataset(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key].unsqueeze(0) for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        return dict(
            input_ids=input_ids,
            labels=labels
        )


@dataclass
class DataCollatorForLMPackDataset(object):

    def __call__(self, instances):
        input_ids, attention_masks = tuple([instance[key].unsqueeze(0) for instance in instances] for key in ["input_ids", "attention_mask"])
        batch_seq_num = instances[0]["labels"][2]
        labels = ([instance["labels"][0].unsqueeze(0) for instance in instances], [instance["labels"][1].unsqueeze(0) for instance in instances])
        input_ids = torch.cat(input_ids, dim=0)
        labels = (torch.cat(labels[0], dim=0), torch.cat(labels[1], dim=0))
        labels = (labels[0], labels[1] * torch.cuda.device_count() / batch_seq_num)
        max_length = input_ids.shape[1]
        attention_mask = attention_masks[0].squeeze()
        acc_length = max_length
        for new_attention_mask in attention_masks[1:]:
            new_attention_mask = new_attention_mask.squeeze()
            attention_mask = torch.cat([attention_mask, new_attention_mask[1:]+acc_length], dim=0)
            acc_length += max_length
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

def make_supervised_data_module(data_args) -> Dict:
    if data_args.batch_method == "naive":
        train_dataset = LMDataset(data_args.train_file)
        data_collator = DataCollatorForLMDataset()
    elif data_args.batch_method == "pack":
        train_dataset = LMPackDataset(data_args.train_file)
        data_collator = DataCollatorForLMPackDataset()
    elif data_args.batch_method == "sort":
        train_dataset = LMSortDataset(data_args.train_file)
        data_collator = DataCollatorForLMDataset()
    return dict(train_dataset=train_dataset, data_collator=data_collator)

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
 

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    seed_everything(42)
    
    config = get_llama_qk(model_name=model_args.model_name_or_path,
                              flash_attention=training_args.flash_attention,
                              pool_size=64,
                              thresh=training_args.thresh)
    
    config = set_RoPE(config, training_args.model_max_length)

    if training_args.bf16:
        model_param_type = torch.bfloat16
    else:
        model_param_type = torch.float32
        
    model = LlamaForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                        torch_dtype=model_param_type,
                                        config=config)
    
    
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
    
    
    if model_args.pack_loss:
        model.pack_loss = True
    data_module = make_supervised_data_module(data_args=data_args)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    for n, p in model.named_parameters():
        if 'predictor.klinear' in n or 'predictor.qlinear' in n:
            p.requires_grad = True
            # torch.nn.init.xavier_uniform_(p)
        else:
            p.requires_grad = False
    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing
    trainer = DynamicPruningPredictorTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model()
    trainer.save_state()

if __name__ == "__main__":
    train()