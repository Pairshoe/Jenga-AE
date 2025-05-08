# Jenga AE

This is the artifact repository for paper #122 at ATC'25, titled **Jenga: Enhancing Long-Context Fine-tuning of LLMs with Contextual Token Sparsity**.

This repository includes:

- **Checkpoints (`checkpoints/`):** The fine-tuned model weights for accuracy validation.
- **Dataset (`dataset/`):** The cleaned E2E dataset used for performance evaluation.
- **Log Files  (`logs/` ):** Experiment logs used for generating the figures in the paper.
- **Reproduced Figures (`output_figures/`):** Output directory of reproduced figures. We have provided figures (prefixed with `exp-`) in this directory that were reproduced for reference.
- **Source Code (`src/`):** The core implementation of Long Exposure.
- **Experiment Scripts (`scripts/`):** Ready-to-use scripts for running experiments corresponding to each figure and table in the paper.

## Installation

We have compiled a list of all the necessary software dependencies and their specified versions in `requirements.txt` except **Flash Attention**. After installing Python (we recommend version 3.10), these dependencies can be installed automatically by executing:

```
pip install -r requirements.txt
```

Then, you need to install **Flash Attention**:

```
pip install flash-attn --no-build-isolation
```


Finally, you can install Jenga from source:

```
pip install -e .
```

## Getting Start

### 1. Quick Reproduction: Plotting from Raw Data

> **Hardware requirements: No GPUs are needed.**
>
> **Estimated Time: about 2 minites.**

To plot all figures in the evaluation section, execute the following command:

```
bash RUNME-a.sh
```

Once you have successfully run this command,figures will be stored in the directory `output_figures/`.

The RUNME-a.sh script reads the original log files, performs some post-processing, and plots the figures. The generated figures will be identical to those in the paper.

The matching relationship between the names of the generated figures and those in the paper is:

| Generated Figure Folder Name | Corresponding Figure in the Paper |
| ---- | ---- |
| end2end/memory | Figure 12 |
| end2end/time | Figure 13|
| ablations/memory-breakdown | Figure 14 (Upper) |
| ablations/time-breakdown | Figure 14 (Lower) |
| ablations/algorithm | Figure 15 |
| ablations/predictor | Figure 16 (Left) |
| extension/2d | Figure 19 (Upper) |
| extension/2d | Figure 19 (Lower) |

**Note:** To reproduce Figure 18, the script will generate two pickle files in the `logs/ablations/segment` directory. Simply drag these files into [memory_viz](https://docs.pytorch.org/memory_viz) to recreate the visualization.


### 2. In-depth Reproduction: Plotting from Actual Run

1. **Figures Reproduction.**

To reproduce all the experiments in the paper, execute the following two commands on corresponding hardware platform:
> **Hardware requirements: 1 NVIDIA A800 GPU.**
>
> **Estimated Time: about ? hours.**


```
bash RUNME-b-a800.sh
```

> **Hardware requirements: 1 NVIDIA A40 GPU.**
>
> **Estimated Time: about ? hours.**


```
bash RUNME-b-a40.sh
```

Once you have successfully run this command, all the figures will be stored in the directory `output_figures/`.

Due to fluctuations in hardware performance, the generated figures may differ slightly from those in the paper.

The matching relationship between the names of the generated figures and those in the paper is the same as the table above.



2. **Table IV Reproduction.(TODO)**

> **Hardware requirements: 1 NVIDIA A100 GPU.**
> 
> **Estimated Time: about 6 hours.**

We provide the fine-tuned model weights in the checkpoints/ directory, allowing you to directly perform inference on downstream tasks for evaluation.

Due to the large size of the model weights, we have split them into multiple sub-units and used Git LFS to upload them to GitHub.

To retrieve the model weights, first execute the following command to pull the large files:

```
git lfs pull
```

Next, run the following scripts to concatenate the sub-units:

```
cd checkpoints
bash cat_tensor.sh
```

We use the framework lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness) from EleutherAI to simplify the evaluation, which can be installed by:

```
pip install lm-evel
```

To evaluate OPT-350M, execute the following command:

```
# Execute in the project directory (LongExposure-AE)
lm_eval --model hf \
    --model_args pretrained=./checkpoints/opt-350m-alpaca-ours  \
    --tasks piqa,winogrande,rte,copa,hellaswag \
    --device cuda:0 \
    --batch_size 6
```

The accuracy of each downstream tasks will output to the console. Adjust the `batch_size` parameter to fit the memory capacity of your device.

Similarly, to evaluate OPT-1.3B and OPT-2.7B, execute the following commands:

```
# Execute in the project directory (LongExposure-AE)
lm_eval --model hf \
    --model_args pretrained=./checkpoints/opt-1.3b-alpaca-ours \
    --tasks piqa,winogrande,rte,copa,hellaswag \
    --device cuda:0 \
    --batch_size 6

lm_eval --model hf \
    --model_args pretrained=./checkpoints/opt-2.7b-alpaca-ours \
    --tasks piqa,winogrande,rte,copa,hellaswag \
    --device cuda:0 \
    --batch_size 6
```

**Note:** In fact, we also provide fine-tuning scripts to obtain these model weights, located in the directory `./src/experiments/overall-accuracy/`. The script `finetune.py` controls the fine-tuning process, while `merge_and_save.py` handles the merging and saving of LoRA weights. However, executing these scripts can take tens of hours. Therefore, we have provided the fine-tuned model weights and included these scripts just for completeness.

3. **Figure 14 Reproduction.**

> **Hardware requirements: 4 NVIDIA A6000 GPUs.**
> 
> **Estimated Time: about 10 minutes.**

We enable distributed model training with the framework DeepSpeed (https://github.com/microsoft/DeepSpeed) from Microsoft, which can be installed by:

```
pip install deepspeed
```

To reproduce Figure 14, execute the following command:

```
bash ./scrips/scale-card/run.sh
```

Once you have successfully run this command, you will get the resulting figure stored in the directory `output_figures/`.

### 3. Detailed Reproduction: Plotting for Each Experiments.

We provide a single script for each experiment in the directory `scripts/`.

The correspondence between the names of the generated figures and those in the paper is detailed in the table above.

To reproduce a specific experiment, locate the corresponding subdirectory and execute the script within it. The resulting figure will be stored in the directory `output_figures/`.
