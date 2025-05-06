from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.mistral.configuration_mistral import MistralConfig


def get_opt_baseline(model_name='facebook/opt-350m',
                     flash_attention=True,
                     use_block=True,
                     use_llora=False,
                     block_ratio=1/4,
                     sparse=0.2):
    config = OPTConfig.from_pretrained(model_name)
    if flash_attention:
        config.attn_implementation = "flash_attention_2"
    else:
        config.attn_implementation = "eager"
    config.use_block = use_block
    config.block_ratio = block_ratio
    config.use_llora = use_llora
    config.sparse = sparse
    return config

def get_llama_baseline(model_name='meta-llama/Meta-Llama-3-8B',
                     flash_attention=True,
                     use_block=True,
                     use_llora=False,
                     block_ratio=1/4,):
    config = LlamaConfig.from_pretrained(model_name)
    if flash_attention:
        config.attn_implementation = "flash_attention_2"
    else:
        config.attn_implementation = "eager"
    config.use_block = use_block
    config.block_ratio = block_ratio
    config.use_llora = use_llora
    return config

def get_llama_qk(model_name='meta-llama/Meta-Llama-3-8B',
                     flash_attention=True,
                     pool_size=64,
                     thresh=0.02,
                     mlp_cut=0.5):
    config = LlamaConfig.from_pretrained(model_name)
    if flash_attention:
        config.attn_implementation = "flash_attention_2"
    else:
        config.attn_implementation = "eager"
    config.pool_size = pool_size
    config.thresh = thresh
    config.sparse = thresh
    config.mlp_cut = mlp_cut
    return config

def get_opt_qk(model_name='facebook/opt-350m',
                     flash_attention=True,
                     pool_size=64,
                     thresh=0.02,
                     mlp_cut=0.5):
    config = OPTConfig.from_pretrained(model_name)
    if flash_attention:
        config.attn_implementation = "flash_attention_2"
    else:
        config.attn_implementation = "eager"
    config.pool_size = pool_size
    config.thresh = thresh
    config.sparse = thresh
    config.mlp_cut = mlp_cut
    return config

def get_mistral_qk(model_name='mistralai/Mistral-7B-v0.1',
                     flash_attention=True,
                     pool_size=64,
                     thresh=0.02,
                     mlp_cut=0.5):
    config = MistralConfig.from_pretrained(model_name)
    if flash_attention:
        config.attn_implementation = "flash_attention_2"
    else:
        config.attn_implementation = "eager"
    config.pool_size = pool_size
    config.thresh = thresh
    config.sparse = thresh
    config.mlp_cut = mlp_cut
    return config

def get_llama_llora(model_name='meta-llama/Meta-Llama-3-8B',
                     flash_attention=True,
                     block_ratio=0.25,
                     ):
    config = LlamaConfig.from_pretrained(model_name)
    if flash_attention:
        config.attn_implementation = "flash_attention_2"
    else:
        config.attn_implementation = "eager"
    config.block_ratio = block_ratio
    return config

def get_opt_llora(model_name='facebook/opt-350m',
                     flash_attention=True,
                     block_ratio=0.25,
                     ):
    config = OPTConfig.from_pretrained(model_name)
    if flash_attention:
        config.attn_implementation = "flash_attention_2"
    else:
        config.attn_implementation = "eager"
    config.block_ratio = block_ratio
    return config
