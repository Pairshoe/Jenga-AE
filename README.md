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

For **in-depth reproduction**, we first need to obtain the model weights. We provide the model weights in the checkpoints/ directory, allowing you to directly perform evaluation.

Due to the large size of the model weights, we use Git LFS to upload them to GitHub.

To retrieve the model weights, first execute the following command to pull the large files:

```
git lfs pull
```
This part of the experiment is conducted on a single GPU, please run:

```
export CUDA_VISIBLE_DEVICES=0
```


1. **Figures Reproduction.**

To reproduce all the experiment figures in the paper, execute the following two commands on corresponding hardware platform:
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

Once you have successfully run these commands, all the figures will be stored in the directory `output_figures/`.

Due to fluctuations in hardware performance, the generated figures may differ slightly from those in the paper.

The matching relationship between the names of the generated figures and those in the paper is the same as the table above.



2. **Tables Reproduction**

> **Hardware requirements: 1 NVIDIA A800 GPU.**
> 
> **Estimated Time: about ? hours.**

To reproduce Table 6,execute the following command:

```
bash scripts/end2end-longbench/run.sh
```
After finishing the script, the **LoneBench prediction and scores** will be stored in the directory `./pred`.

To reproduce Table 7,execute the following command:

```
bash scripts/end2end-ppl/ppl.sh
```
After finishing the script, the **Perplexity (PPL) results** will be stored in the directory `logs/end2end/accuracy/`.



