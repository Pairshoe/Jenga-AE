# [ATC'25 Artifact] Jenga: Enhancing Long-Context Fine-tuning of LLMs with Contextual Token Sparsity

This is the artifact repository for submission #122 at ATC'25, titled *Jenga: Enhancing Long-Context Fine-tuning of LLMs with Contextual Token Sparsity*.

Should there by any questions, please contact the authors in HotCRP. The authors will respond to each question within 24 hours and as soon as possible.

## Repository Contents

We provide all necessary components—code, scripts and logs—to fully reproduce the results presented in the paper. Specifically:

- **Model and Predictor Weights  (`checkpoints/` ):** Pre-trained model weights and predictor weights for the experiments in the paper.
- **Dataset (`dataset/` ):** Datasets used in the experiments.
- **Log Files  (`logs/` ):** Experiment logs used for figure generation in the paper.
- **Output Figures (`output_figures/`):** Generated figures from the logs.
- **Experiment Scripts (`scripts/`):** Ready-to-use scripts for running experiments corresponding to each figure and table in the paper.
- **Source Code (`src/`):** Core implementation of our system.

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

## Data Preparation

Since the model weights and datasets are distributed across different sources, we have listed the download links below to simplify the reproduction process. To further improve the efficiency of running the AE, we also provide fine-tuned weights, as the fine-tuning process can be time-consuming.

### 1. Model Weights

Please download the model weights `peft_model.zip` and `predictor.zip` from [Models Link](https://cloud.tsinghua.edu.cn/d/29c75b36a8d94016a402/), unzip the folders and place them in `./checkpoints` directory.

After that, please download the following models from Hugging Face and place them in the specified path as shown below.

| Destination Path      | Source URL     |
|-------------------------|------------------------------------|
|checkpoints/llama3|[meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)|
|checkpoints/llama2|[meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)|
|checkpoints/opt-6.7b|[facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b)|
|checkpoints/opt-2.7b|[facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b)|
|checkpoints/opt-1.3b|[facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b)|
|checkpoints/opt-350m|[facebook/opt-350m](https://huggingface.co/facebook/opt-350m)|
|checkpoints/opt-125m|[facebook/opt-125m](https://huggingface.co/facebook/opt-125m)|

**Note:** You need to apply for access before downloading the LLaMA 2 or LLaMA 3 models.

The ideal file structure is shown below. 

```
checkpoints/
├── llama2/
├── llama3/
├── opt-6.7b/
├── opt-2.7b/
├── opt-1.3b/ 
├── opt-350m/
├── opt-125m/
├── peft_model/
└── predictor/
```

### 2. Datasets

Please download the datasets `dataset.zip` from [Datasets Link](https://cloud.tsinghua.edu.cn/f/d7ac0d22f4ec4faaa547/?dl=1), unzip the folder and place it in the **project root directory**.

The ideal file structure is shown below. 

```
dataset/
├── LongAlign/
├── PPL/
├── RedPajama-Data-1T-Sample/
└── longbench/
```

## Getting Start

We provide three scripts tailored to different user needs to help you get started with our project:

- **Environment Setup Verification**: This simple script ensures that your environment is correctly configured, and that all basic components are running smoothly. It checks if all dependencies are installed and functioning as expected.
- **Quick Reproduction**: This script allows you to quickly reproduce the figures from our paper using preprocessed data, without requiring a GPU. It’s ideal for users looking for a fast demonstration of the results.
- **In-Depth Reproduction**: This script is designed for users who wish to run the full evaluation with the original model weights, enabling the exact reproduction of the results presented in the paper.

### 1. Hello-world Example: Environment Setup Verification

To verify that everything is correctly set up for Jenga, including model files, datasets, and environment compatibility, please run the following command:

```
bash hello_world.sh
```

It will run for approximately 10 seconds and, on success, output something like below:

```
======================================================
 Welcome to the Jenga Environment Setup Checker!
======================================================

--- Checking for base model files ---
All base model configuration files seem to be present.

--- Checking for PEFT artifacts ---
All checked PEFT artifacts seem to be present.

--- Checking for datasets ---
All datasets seem to be present.

--- Running environment compatibility and Jenga functionality test ---
  Loading LlamaForCausalLM model...
  Loading predictor attention state dictionary...
  Performing a test forward and backward pass...
  Environment compatibility and Jenga functionality test PASSED.

------------------------------------------------------
Congratulations! All checks passed. Your Jenga environment appears to be set up correctly.
======================================================

```
If you see this output, everything is ready and you can proceed with running Jenga.

### 2. Quick Reproduction: Plotting from Raw Data

> **Hardware requirements: No GPUs are needed.**
>
> **Estimated Time: about 2 minites.**

To plot all figures in the evaluation section, execute the following command:

```
bash RUNME-a.sh
```

Once you have successfully run this command, figures will be stored in the directory `output_figures/`.

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

We provide two types of scripts in this section:

- **One-step Execution:** These scripts reproduce all the results from the paper in a single run. This approach offers convenience but comes with a significant time cost. 
 
- **Separate Execution:** These scripts allow you to selectively reproduce the experiments you are interested in, which is more flexible.

**Note:** Before getting started, please run the command below because the following experiments are conducted on a single GPU.

```
export CUDA_VISIBLE_DEVICES=0
```

#### 3.1 One-step Execution

> **Hardware requirements: 1 NVIDIA A800 GPU or 1 NVIDIA A40 GPU.**
>
> **Estimated Time: about 5 hours.**

**(1) Figure Reprodution**

To reproduce all the experiment figures in the paper, execute the following two commands on corresponding hardware platform:
```
bash RUNME-b-a800.sh  # Hardware requirements: 1 NVIDIA A800 GPU

bash RUNME-b-a40.sh   # Hardware requirements: 1 NVIDIA A40 GPU
```

Once you have successfully run these commands, all the figures will be stored in the directory `output_figures/`.

Due to fluctuations in hardware performance, the generated figures may differ slightly from those in the paper.

The matching relationship between the names of the generated figures and those in the paper is the same as the table above.

**(2) Table Reproduction**

To reproduce Table 6,execute the following command:

```
bash scripts/end2end-longbench/run.sh   # Hardware requirements: 1 NVIDIA A800 GPU 
```
After finishing the script, the **LoneBench prediction and scores** will be stored in the directory `./logs/end2end/accuracy/longbench`.

To reproduce Table 7, execute the following command:

```
bash scripts/end2end-ppl/ppl.sh  # Hardware requirements: 1 NVIDIA A800 GPU
```
After finishing the script, the **Perplexity (PPL) results** will be stored in the directory `logs/end2end/accuracy/`.

#### 3.2 Separate Execution

The reproduction scripts for each experiment and their expected execution time are summarized in the table below.  

You can reproduce the experiments seperately based on your interests.

| Generated Output Folder Name | Corresponding Figure/Table in the Paper | Script Path  | Expected Runtime |
|-----------------------------|-------------------|---------------------------------------|----------|
| output_figures/end2end/memory              | Figure 12         | scripts/end2end-memory/run.sh         | 40 mins  |
| output_figures/end2end/time-a800                | Figure 13         | scripts/end2end-time/run.sh (A800) <br>scripts/end2end-time/run-a40.sh (A40)| 50 mins  |
| output_figures/ablations/memory-breakdown | Figure 14 (Upper) | scripts/ablation-mem-breakdown/run.sh | 10 mins  |
| output_figures/ablations/time-breakdown   | Figure 14 (Lower) | scripts/ablation-time-breakdown/run.sh| 10 mins  |
| output_figures/ablations/algorithm        | Figure 15         | scripts/ablation-algorithm/run.sh     | 5 mins  |
| output_figures/ablations/predictor        | Figure 16 (Left)  | scripts/ablation-predictor/run.sh     | 3 hours  |
| output_figures/ablations/segment          | Figure 18 (Drag to [memory_viz](https://docs.pytorch.org/memory_viz)) | scripts/ablation-segment/run.sh           | 5 mins  |
| output_figures/extension/2d               | Figure 19 (Upper) | scripts/extension-2d/run.sh           | 25 mins  |
| output_figures/extension/offload          | Figure 19 (Lower) | scripts/extension-offload/run.sh      | 30 mins  |
| logs/end2end/accuracy/longbench          | Table 6 | scripts/end2end-longbench/run.sh      | 3 hours  |
| logs/end2end/accuracy          | Table 7 | scripts/end2end-ppl/ppl.sh      | 3 hours  |
