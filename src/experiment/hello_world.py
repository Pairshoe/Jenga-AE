import contextlib  # For redirecting stdout
import math
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import \
    AutoTokenizer  # Retained: Although tokenize_fn is removed, AutoTokenizer might be implicitly needed by from_pretrained or useful for users extending this script. Given it was there, and without full context of jenga's needs for from_pretrained with a path, it's safer to keep if a tokenizer might be loaded separately. Let's assume for now it might be needed by underlying Jenga calls or for debugging. If truly unused by Jenga's from_pretrained with a path, it can be removed.
from transformers import logging as hf_logging

from jenga.models.modeling_llama import LlamaForCausalLM
from jenga.utils.config_utils import get_llama_qk

# --- Configuration Constants ---
BASE_CHECKPOINT_DIR = "checkpoints"
DATASET_DIR = "dataset"

# Model configurations: (name, path_fragment, download_source_hint)
MODEL_CONFIGS = [
    ("llama2", "llama2/config.json", "the official Llama resources or your cloud driver"),
    ("llama3", "llama3/config.json", "the official Llama resources or your cloud driver"),
    ("opt-350m", "opt-350m/config.json", "Hugging Face"),
    ("opt-1.3b", "opt-1.3b/config.json", "Hugging Face"),
    ("opt-2.7b", "opt-2.7b/config.json", "Hugging Face"),
    ("opt-6.7b", "opt-6.7b/config.json", "Hugging Face"),
]

PEFT_METHODS = ["jenga", "lora"]
PEFT_SEQ_LENS = ["8k", "10k", "12k", "14k", "16k"]

DATASET_NAMES = ["16384", "longbench", "RedPajama-Data-1T-Sample", "test_pg19.bin", "proof_pile.bin"]


def check_base_models_exist() -> bool:
    """Checks if all required base model configuration files exist."""
    print("\n--- Checking for base model files ---")
    all_found = True
    for model_name, path_fragment, source_hint in MODEL_CONFIGS:
        model_config_path = os.path.join(BASE_CHECKPOINT_DIR, path_fragment)
        if not os.path.exists(model_config_path):
            print(f"ERROR: Base model '{model_name}' config not found at: {model_config_path}")
            print(f"       Please download '{model_name}' from {source_hint}.")
            all_found = False
    
    if all_found:
        print("All base model configuration files seem to be present.")

    return all_found

def check_peft_artifacts_exist() -> bool:
    """Checks for PEFT related model artifacts and predictors."""
    print("\n--- Checking for PEFT artifacts ---")
    all_found = True
    
    predictor_path = os.path.join(BASE_CHECKPOINT_DIR, "predictor/predictor.pth")
    if not os.path.exists(predictor_path):
        print(f"ERROR: Predictor weights not found at: {predictor_path}")
        all_found = False


    # Check for adapter_model.safetensors
    for method in PEFT_METHODS:
        # Check la/*
        la_adapter_path = os.path.join(BASE_CHECKPOINT_DIR, f"peft_model/la/{method}/adapter_model.safetensors")
        if not os.path.exists(la_adapter_path):
            print(f"ERROR: PEFT LA adapter for '{method}' not found at: {la_adapter_path}")
            all_found = False

            
        # Check rp/*/*
        for seq_len in PEFT_SEQ_LENS:
            rp_adapter_path = os.path.join(BASE_CHECKPOINT_DIR, f"peft_model/rp/{seq_len}/{method}/adapter_model.safetensors")
            if not os.path.exists(rp_adapter_path):
                print(f"ERROR: PEFT RP adapter for '{method}' (seq_len: {seq_len}) not found at: {rp_adapter_path}")
                all_found = False

                
    if  all_found:
        print("All checked PEFT artifacts seem to be present.")
    else:
        print("The missing PEFT artifacts should be downloaded from the cloud driver.")
    return all_found

def check_datasets_exist() -> bool:
    """Checks if the specified datasets exist."""
    print("\n--- Checking for datasets ---")
    all_found = True
    for dataset_name in DATASET_NAMES:
        dataset_path = os.path.join(DATASET_DIR, dataset_name)
        if not os.path.exists(dataset_path):
            print(f"ERROR: Dataset '{dataset_name}' not found at: {dataset_path}")
            all_found = False

            
    if all_found:
        print("All datasets seem to be present.")
    else:
        print("The missing datasets should be downloaded from the cloud driver.")

    return all_found

def run_environment_compatibility_test() -> bool:
    """
    Runs a quick test to verify PyTorch, Jenga core functionalities,
    and CUDA environment compatibility.
    """
    print("\n--- Running environment compatibility and Jenga functionality test ---")
    
    # Helper function for RoPE scaling (specific to this test)
    def set_RoPE(config, model_max_length: int):
        orig_rope_scaling = getattr(config, "rope_scaling", None)
        if orig_rope_scaling is None:
            orig_rope_scaling = {"factor": 1.0} # Ensure factor is float

        orig_rope_scaling_factor = orig_rope_scaling.get("factor", 1.0)
        orig_ctx_len = getattr(config, "max_position_embeddings", None)

        if orig_ctx_len:
            # Ensure orig_ctx_len is an int or float before multiplication
            if not isinstance(orig_ctx_len, (int, float)):
                # print(f"Warning: max_position_embeddings ('{orig_ctx_len}') is not a number. RoPE scaling might be incorrect.")
                return config # or handle error appropriately

            effective_orig_ctx_len = orig_ctx_len * orig_rope_scaling_factor
            if model_max_length > effective_orig_ctx_len:
                scaling_factor = float(math.ceil(model_max_length / effective_orig_ctx_len))
                config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        return config

    try:
        # print("  Initializing Jenga model configuration...")
        # This test relies on 'llama2' checkpoints and 'predictor' being available.
        # These should ideally be confirmed by previous checks.
        hf_logging.set_verbosity_error()
        
        llama2_model_path = os.path.join(BASE_CHECKPOINT_DIR, "llama2")
        predictor_config_path = os.path.join(BASE_CHECKPOINT_DIR, "predictor", "pruned_config.pth")
        predictor_weights_path = os.path.join(BASE_CHECKPOINT_DIR, "predictor", "predictor.pth")

        if not os.path.exists(os.path.join(llama2_model_path, "config.json")):
            print(f"  ERROR: Llama2 model config for test not found at '{llama2_model_path}'. Cannot proceed with Jenga functionality test.")
            return False
        if not os.path.exists(predictor_config_path) or not os.path.exists(predictor_weights_path):
            print(f"  ERROR: Predictor files for test not found. Searched for '{predictor_config_path}' and '{predictor_weights_path}'. Cannot proceed.")
            return False

        config = get_llama_qk(model_name=llama2_model_path,
                              flash_attention=True, # Assumes FlashAttention is available or Jenga handles its absence
                              pool_size=64,
                              thresh=0.4)
        config = set_RoPE(config, 4096) # Example sequence length for testing
        
        pruned_cfg = torch.load(predictor_config_path, map_location='cpu')
        layers_cfg = pruned_cfg["layers"]
        config.predictor_layers = layers_cfg
        
        print("  Loading LlamaForCausalLM model...")
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            model = LlamaForCausalLM.from_pretrained(
                llama2_model_path,
                torch_dtype=torch.bfloat16, # Ensure your hardware supports bfloat16
                config=config
            )
      
        print("  Loading predictor attention state dictionary...")
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            attn_state_dict = torch.load(predictor_weights_path, map_location='cpu')
            
            loaded_keys_count = 0
            model_s_dict = model.state_dict()
            for k, v in attn_state_dict.items():
                if k in model_s_dict:
                    model_s_dict[k].copy_(v)
                    loaded_keys_count += 1
                
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Ensure these modules exist in the model
                lora_dropout=0.0, # Explicitly float
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
        
        
        if not torch.cuda.is_available():
            print("  ERROR: CUDA is not available. This test requires a CUDA-enabled GPU.")
            return False
        
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            model = model.to('cuda')
            model.train() # Set model to training mode for the test
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            
            # Use model's vocab size for dummy data if available, otherwise a common default
            vocab_size = getattr(model.config, "vocab_size", 32000)
            input_ids = torch.randint(0, vocab_size, (1, 4096), device='cuda')
            labels = torch.randint(0, vocab_size, (1, 4096), device='cuda')
                
        print("  Performing a test forward and backward pass...")
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): # Mixed precision
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() # Good practice
        
        print("  Environment compatibility and Jenga functionality test PASSED.")
        return True

    except ImportError as e:
        print(f"  ERROR: An import error occurred: {e}")
        print("         Please ensure all dependencies (PyTorch, Transformers, PEFT, Jenga, etc.) are correctly installed.")
        return False
    except Exception as e:
        import traceback
        print(f"  ERROR: An unexpected error occurred during the test: {e}")
        print("  Traceback:")
        traceback.print_exc()
        print("         Please review your installation, model files, and environment configuration.")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    print("======================================================")
    print(" Welcome to the Jenga Environment Setup Checker!")
    print("======================================================")
    
    overall_status_ok = True
    
    if not check_base_models_exist():
        overall_status_ok = False
        
    if not check_peft_artifacts_exist():
        overall_status_ok = False
        
    if not check_datasets_exist():
        overall_status_ok = False
        
    # Only run the more intensive compatibility test if basic file checks pass
    if overall_status_ok:
        if not run_environment_compatibility_test():
            overall_status_ok = False
    
    print("\n------------------------------------------------------")
    if overall_status_ok:
        print("Congratulations! All checks passed. Your Jenga environment appears to be set up correctly.")
    else:
        print("Setup Incomplete: One or more checks failed. Please review the error messages above and take corrective actions.")
    print("======================================================")