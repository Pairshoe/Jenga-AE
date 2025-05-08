import os
import sys

path_to_check = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if path_to_check not in sys.path:
    sys.path.append(path_to_check)


from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
import transformers
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from jenga.trainer.time_profile  import Trainer

from jenga.models.modeling_opt_base import OPTForCausalLM, OPTLearnedPositionalEmbedding
from jenga.utils.config_utils import get_opt_baseline
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
    model_name_or_path: Optional[str] = field(default="../models/opt-350m")
    pack_loss: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    flash_attention: bool = field(default=True)
    model_max_length: int = field(default=16384)
    gradient_checkpoint : bool = field(default=False)


 

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
    
    if  training_args.model_max_length % 2048 != 0:
        training_args.model_max_length = 2048 * (training_args.model_max_length // 2048 + 1)
    
    config = get_opt_baseline(model_name=model_args.model_name_or_path,
                              flash_attention=training_args.flash_attention,)
    
    if training_args.bf16:
        model_param_type = torch.bfloat16
    else:
        model_param_type = torch.float32
        
    model = OPTForCausalLM.from_pretrained(model_args.model_name_or_path,
                                           torch_dtype=model_param_type,
                                           config=config,)  
    #pos embedding
    max_positions = training_args.model_max_length - 2
    original_num_embeddings = model.model.decoder.embed_positions.num_embeddings - 2
    print(f"original_num_embeddings: {original_num_embeddings}")
    # assert (max_positions + 2) % original_num_embeddings == 0
    original_embed_positions = model.model.decoder.embed_positions.weight.data
    print(f"original_embed_positions: {original_embed_positions.shape}")
    assert (max_positions + 2) % original_num_embeddings == 0
    duplicated_embed_positions =  torch.cat([original_embed_positions[:-2] * i for i in range(1, (max_positions + 2) // original_num_embeddings + 1)] +  [original_embed_positions[-2:]], dim = 0)
    model.model.decoder.embed_positions = OPTLearnedPositionalEmbedding((max_positions + 2), model.model.decoder.embed_positions.embedding_dim)
    model.model.decoder.embed_positions.weight.data = duplicated_embed_positions
    
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
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    if training_args.gradient_checkpoint:
        model.config.use_cache = False         # required for gradient checkpointing
        model.enable_input_require_grads()     # required for gradient checkpointing
        model.gradient_checkpointing_enable()  # enable gradient checkpointing
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