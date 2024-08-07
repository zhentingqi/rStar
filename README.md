# rStar

##  Generator

This repository contains scripts to run the  dataset generator using our custom method.

### Prerequisites

- Python 3.10
- NVIDIA GPU
- Necessary Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

To run the  generator, use the following command:

```sh
CUDA_VISIBLE_DEVICES=0 python run_src/ours/run_ours.py \
    --mode run \
    --method ours \
    --dataset_name  \
    --test_json_filename test_all \
    --mcts_reward_mode last_only \
    --enable_tot \
    --model_ckpt /path/to/model \
    --note default \
    --answer_selection_metric select_response \
    --answer_selection_mode topk \
    --topk 1 \
    --num_rollouts 16
```

### Config
The script run_gsm8k_generator.sh includes several configurable parameters:
```txt
--mode: Mode to run the script (default: run).

--method: Method to use (default: ours).

--dataset_name: Name of the dataset (default: ).

--test_json_filename: Filename for the test JSON (default: test_all).

--mcts_reward_mode: MCTS reward mode (default: last_only).

--enable_tot: Enable TOT (default: enabled).

--model_ckpt: Path to the model checkpoint.

--note: Additional notes (default: default).

--answer_selection_metric: Metric for answer selection (default: select_response).

--answer_selection_mode: Mode for answer selection (default: topk).

--topk: Top K answers to select (default: 1).

--num_rollouts: Number of rollouts (default: 16).
```
Make sure to adjust these parameters according to your requirements.

##  Discriminator

This repository contains scripts to run the  discriminator using our custom method.

### Usage

To run the  discriminator, use the following command:

```sh
./scripts/run_gsm8k_discriminator.sh <CUDA_DEVICE> <DATASET_NAME> <ROOT_DIR>
```

### Config
The script run_gsm8k_discriminator.sh includes several configurable parameters:

METHOD: Method to use (default: majvote).

BASE_ROOT: Base root directory (default: empty).

CUDA: CUDA device to use (passed as the first argument).

DATASET_NAME: Name of the dataset (passed as the second argument).

ROOT_DIR: Root directory (passed as the third argument).

PROMPT_DIR: Directory for prompts (default: prompts).

The Python script run_src/ours/do_discriminate.py includes additional parameters:

```txt
--model_ckpt: Path to the model checkpoint.

--method: Method to use (default: majvote).

--root_dir: Root directory.

--prompt_dir: Directory for prompts.

--dataset_name: Name of the dataset.

--mask_left_boundary: Left boundary for masking (default: 0.5).

--mask_right_boundary: Right boundary for masking (default: 0.5).

--num_masked_solution_traces: Number of masked solution traces (default: 1).

--rc_mode: RC mode (default: maj).

--rc_temperature: RC temperature (default: 0).

--rc_n_completions: Number of RC completions (default: 10).

--threshold: Threshold value (default: 0.999).

--note: Additional notes (default: default).

--generator: Generator to use (default: ours).

--add_solution_trace: Add solution trace flag.
```