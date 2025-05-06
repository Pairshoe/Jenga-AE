import os
import sys
import torch
from tqdm import tqdm
from functools import partial
import argparse
import transformers
import math
import json

path_to_check = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if path_to_check not in sys.path:
    sys.path.append(path_to_check)

from datasets import Dataset, DatasetDict, load_dataset
from echo.predictor.attention_predictor import AttnPredictor
from echo.utils.config_utils import get_llama_config
from echo.utils.profile_utils import get_train_input, get_train_label
from echo.models.llama_train_predictor import LlamaForCausalLM
from echo.utils.others import (
    get_shift_heads_idx,
    load_jsonl,
    smart_tokenizer_and_embedding_resize,
    tokenize_fn,
    seed_everything
)

from torch.utils.data import DataLoader, Randomechopler
from transformers.trainer_utils import seed_worker
from transformers import DataCollatorForLanguageModeling

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"   #默认填充
DEFAULT_EOS_TOKEN = "</s>"    #句子结束
DEFAULT_BOS_TOKEN = "<s>"       #句子开始
DEFAULT_UNK_TOKEN = "<unk>"     #未知

parser = argparse.ArgumentParser(description='train predictor')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--predictor_lr', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--steps', type=int, default=300)
parser.add_argument('--model_name', type=str, default="../models/llama")
parser.add_argument('--dataset', type=str, default="gov_report")
parser.add_argument('--flash_attention', type=bool, default=False, help='Whether use flash attention for training')
parser.add_argument('--seq_len', type=int, default=8192, help='The ratio of block size')
parser.add_argument('--block_ratio', type=float, default=0.0625, help='The ratio of block size')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, input_hidden, labels, n_layer):
        self.input_hidden = input_hidden
        self.labels = labels
        self.len_data = len(input_hidden)

        self.processed_labels = []
        for i in range(self.len_data):
            # 假设 labels[j][i] 的维度是 (1, n_head, n_block, n_block)
            layer_labels = torch.stack([labels[j][i].squeeze(0) for j in range(n_layer)], dim=0)
            self.processed_labels.append(layer_labels)

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):
        return self.input_hidden[idx].float(), self.processed_labels[idx]


def make_symmetric(labels):
    # 获取下三角矩阵，包括对角线
    tril_matrix = torch.tril(labels)
    # 将下三角矩阵复制到上三角矩阵位置，保持对角线不变
    symmetric_labels = tril_matrix + tril_matrix.transpose(-1, -2) - torch.diag_embed(torch.diagonal(labels, dim1=-2, dim2=-1))
    return symmetric_labels


def evaluate(attn_predictor, test_dataloader, args, n_layer, n_block, block_size):
    total_correct = 0
    total_echoples = 0

    for predictor_layer in attn_predictor:
        for predictor_block in predictor_layer:
            predictor_block.eval()

    with torch.no_grad():
        for batch_input, batch_label in tqdm(test_dataloader, desc='Evaluating', unit='batch'):
            batch_input = batch_input.to(args.device)
            batch_label = batch_label.to(args.device)
            batch_size = batch_input.size(0)

            for layer in range(n_layer):
                inputs = batch_input
                labels = batch_label[:, layer, :, :, :].float()  # (batch_size, n_head, n_block, n_block)
                all_labels = make_symmetric(labels)  # (batch_size, n_head, n_block, n_block)
                for block in range(n_block):
                    input = inputs[:, block*block_size:(block+1)*block_size]
                    output = attn_predictor[layer][block](input)
                    # 如果模型输出未经过Sigmoid激活，需要在这里添加
                    # output = torch.sigmoid(output)
                    output = output.view(-1, n_head, n_block)
                    predictions = (output >= 0.5).float()
                    correct = (predictions == all_labels[:, :, block, :]).sum().item()
                    total_correct += correct
                    total_echoples += predictions.numel()

    accuracy = total_correct / total_echoples
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

def data_collator(features):
    input_ids = [torch.tensor(f["input_ids"][0], dtype=torch.long) for f in features]
    labels = [torch.tensor(f["labels"][0], dtype=torch.long) for f in features]
    attention_masks = [torch.ones(len(f["input_ids"][0]), dtype=torch.long) for f in features]
    
    # 将序列填充到 8192
    max_length = 8192
    
    # 如果序列长度小于 8192，进行填充
    batch = {
        "input_ids": torch.stack([
            torch.cat([ids, torch.full((max_length - len(ids),), tokenizer.pad_token_id, dtype=torch.long)])
            if len(ids) < max_length else ids[:max_length]
            for ids in input_ids
        ]),
        "labels": torch.stack([
            torch.cat([lbls, torch.full((max_length - len(lbls),), -100, dtype=torch.long)])
            if len(lbls) < max_length else lbls[:max_length]
            for lbls in labels
        ]),
        "attention_mask": torch.stack([
            torch.cat([mask, torch.zeros(max_length - len(mask), dtype=torch.long)])
            if len(mask) < max_length else mask[:max_length]
            for mask in attention_masks
        ]),
    }
    
    return batch
    
if __name__ == '__main__':
    args = parser.parse_args()

    #prepare predcitor

    attn_predictor = []
    optimizers = []
    
    #prepare model
    model_name = args.model_name.split("/")[-1]

    config = get_llama_config(model_name=args.model_name,
                              flash_attention=args.flash_attention,
                              use_block=False,
                              block_ratio=args.block_ratio,
                              ) 
    # Set RoPE scaling factor
    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}

    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if args.seq_len > orig_ctx_len:
            scaling_factor = float(math.ceil(args.seq_len / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    seq_len = args.seq_len
    n_layer = config.num_hidden_layers
    n_head = config.num_attention_heads
    n_block = int(4 / args.block_ratio)
    block_size = seq_len // n_block
    
    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        cache_dir='./cache',
    )  
    
    #prepare tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir="./tmp",
        model_max_length=args.seq_len,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    tokenizer.add_tokens(["[INST]", "[/INST]"])
    #增加特殊符号
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    #prepare data
    dataset2prompt = json.load(open("data/dataset2prompt.json", "r"))
    prompt_format = dataset2prompt[args.dataset]
    dataset = load_dataset('json',data_files=f"./data/llama/{args.dataset}.jsonl")
    # dataset = load_dataset('json', data_files="data_new/gov_report.jsonl")
    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(
        partial(tokenize_fn,tokenizer,prompt_format),
        batched=False,
        remove_columns=train_dataset.column_names,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size = args.batch_size ,
        collate_fn = data_collator,
        num_workers = 0,
        pin_memory = True,
        persistent_workers = False,
        drop_last = False,
        echopler = Randomechopler(train_dataset),
        worker_init_fn=seed_worker,
    )
    
    for i in range(n_layer):
        layer_predictor, layer_optimizer = [], []
        for j in range(n_block):
            predictor = AttnPredictor(block_size=block_size,input_dim=config.hidden_size, hidden_dim=128, output_dim=n_head * n_block)
            predictor = predictor.to(torch.bfloat16)
            predictor = predictor.to(args.device).train()

            optimizer = torch.optim.Adam(predictor.parameters(), lr=args.predictor_lr)

            layer_predictor.append(predictor)
            layer_optimizer.append(optimizer)
        attn_predictor.append(layer_predictor)
        optimizers.append(layer_optimizer)

    # 使用 BCEWithLogitsLoss，如果模型输出未经过Sigmoid
    loss_fn = torch.nn.BCEWithLogitsLoss()

    losses = []
    for i in range(n_layer):
        layer_losses = []
        for j in range(n_block):
            layer_losses.append([])
        losses.append(layer_losses)


    model = model.to(torch.bfloat16)
    model.to('cuda:0')
    model.eval() 
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for step, inputs in enumerate(tqdm(train_dataloader, desc="Training Progress")):
            if step == args.steps:
                break
            inputs = {k: v.to('cuda:0') for k,v in inputs.items()}

            get_train_input.clear()
            get_train_label.clear()
            with torch.no_grad():
                model(**inputs)
            loss_sum = 0.0
            # batch_input: (batch_size, seq_len)
            # batch_label: (batch_size, n_layer, n_head, n_block, n_block)
            batch_input = get_train_input
            batch_label = get_train_label

            for layer in range(n_layer):
                inputs = batch_input[layer][0].float().to(args.device)
                labels = batch_label[layer][0].float().to(args.device)  # (batch_size, n_head, n_block, n_block)
                all_labels = make_symmetric(labels)  # (batch_size, n_head, n_block, n_block)
                for block in range(n_block):
                    input = inputs[:, block * block_size:(block + 1) * block_size]
                    optimizers[layer][block].zero_grad()
                    input = torch.squeeze(input, 0)
                    output = attn_predictor[layer][block](input)
                    output = output.view(-1, n_head, n_block)

                    loss = loss_fn(output, all_labels[:, :, block, :])
                    loss.backward()
                    optimizers[layer][block].step()

                    losses[layer][block].append(loss.item())
                    loss_sum += loss.item()

            print(f"current Loss: {loss_sum/(n_layer*n_block):.4f}")

    # 保存模型
    save_path = f"experiment/train_predictor/model-{args.dataset}.pt"
    torch.save({'attn_predictor_state_dict': [[predictor.state_dict() for predictor in layer] for layer in attn_predictor]}, save_path)

    # # 加载测试数据
    # test_input_hidden = torch.load("experiment/train_predictor/data/input/val.pt")
    # test_labels = torch.load("experiment/train_predictor/data/label/val.pt")

    # test_dataset = MyDataset(test_input_hidden, test_labels, n_layer)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # # 评估模型
    # evaluate(attn_predictor, test_dataloader, args, n_layer, n_block, block_size)
