# [ATC'25 Artifact] Jenga: Enhancing Long-Context Fine-tuning of LLMs with Contextual Token Sparsity

This is the artifact repository for submission #122 at ATC'25, titled *Jenga: Enhancing Long-Context Fine-tuning of LLMs with Contextual Token Sparsity*.

Should there by any questions, please contact the authors in HotCRP. The authors will respond to each question within 24 hours and as soon as possible.

## Repository Contents

We provide all necessary components—code, scripts and logs—to fully reproduce the results presented in the paper. Specifically:

- **Log Files  (`logs/` ):** Experiment logs used for figure generation in the paper.
- **Source Code (`src/`):** Core implementation of our system.
- **Experiment Scripts (`scripts/`):** Ready-to-use scripts for running experiments corresponding to each figure and table in the paper.

## Installation

We have listed all required software dependencies (with specified versions) in `requirements.txt`, except for *Flash Attention*, which should be installed separately due to build constraints.

After installing Python (we recommend version 3.10), install *the required dependencies* by running:

```
pip install -r requirements.txt
```

Next, install *Flash Attention* separately:

```
pip install flash-attn --no-build-isolation
```

Finally, install *Jenga* from source:

```
pip install -e .
```

## Getting Start

We provide three scripts tailored to different user needs to help you get started with our project:

- **Environment Setup Verification**: This simple script ensures that your environment is correctly configured, and that all basic components are running smoothly. It checks if all dependencies are installed and functioning as expected.
- **Quick Reproduction**: This script allows you to quickly reproduce the figures from our paper using preprocessed data, without requiring a GPU. It’s ideal for users looking for a fast demonstration of the results.
- **In-Depth Reproduction**: This script is designed for users who wish to run the full evaluation with the original model weights, enabling the exact reproduction of the results presented in the paper.

### 1. Hello-world Example: Environment Setup Verification

TODO: add descriptions

### 2. Quick Reproduction: Plotting from Raw Data

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
| extension/offload | Figure 19 (Lower) |

**Note:** To reproduce Figure 18, the script will generate two pickle files in the `logs/ablations/segment` directory. Simply drag these files into [memory_viz](https://docs.pytorch.org/memory_viz) to recreate the visualization.


### 3. In-depth Reproduction: Plotting from Actual Run

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



