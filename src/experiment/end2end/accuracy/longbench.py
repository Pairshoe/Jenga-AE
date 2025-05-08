import argparse
import json
import math
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from jenga.models.modeling_llama_base import LlamaForCausalLM
from jenga.utils.config_utils import get_llama_baseline


BEGIN_TOKEN, END_TOKEN = '<<BEGIN>>', '<<END>>'
DEFAULT_PAD_TOKEN = "[PAD]"   #默认填充
DEFAULT_EOS_TOKEN = "</s>"    #句子结束
DEFAULT_BOS_TOKEN = "<s>"       #句子开始
DEFAULT_UNK_TOKEN = "<unk>"     #未知
IGNORE_INDEX = -100


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--model_max_length', type=int, default=32768)
    parser.add_argument('--base_model', type=str, default="checkpoints/llama2")
    parser.add_argument('--baseline', action='store_true', help="Use baseline model")
    parser.add_argument('--peft_model', type=str, default="checkpoints/peft_model/la/jenga")
    return parser.parse_args(args)

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

# This is the customized building prompt for chat models
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

def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path, args):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(args, device)
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(args, device):
    
    config = get_llama_baseline(model_name=args.base_model,
                              flash_attention=True,)

    config = set_RoPE(config, args.model_max_length)
    
    model = LlamaForCausalLM.from_pretrained(args.base_model, 
                                        torch_dtype=torch.bfloat16,
                                        config=config)
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, 
                                                model_max_length=args.model_max_length,
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

    if not args.baseline:
        model = PeftModel.from_pretrained(
            model,
            args.peft_model,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    model = model.to(device)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("dataset/longbench/config/model2path.json", "r"))
    model2maxlen = json.load(open("dataset/longbench/config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.peft_model.split('/')[-1]
    # define your model
    max_length = args.model_max_length
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("dataset/longbench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("dataset/longbench/config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if args.e:
            data = load_dataset('json', data_files={'test': f'./dataset/longbench/{dataset}_e.jsonl'}, split='test').select(range(50))
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('json', data_files={'test': f'./dataset/longbench/{dataset}.jsonl'}, split='test').select(range(50))
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model2path, out_path, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()