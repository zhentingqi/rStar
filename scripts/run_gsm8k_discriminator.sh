METHOD=majvote

BASE_ROOT=""
CUDA=$1
DATASET_NAME=$2
ROOT_DIR=$3

PROMPT_DIR="prompts"

export CUDA_VISIBLE_DEVICES=${CUDA}

python run_src/ours/do_discriminate.py \
    --model_ckpt /path/to/model \
    --method ${METHOD} \
    --root_dir ${BASE_ROOT}${ROOT_DIR} \
    --prompt_dir ${PROMPT_DIR} \
    --dataset_name ${DATASET_NAME} \
    --mask_left_boundary 0.5 \
    --mask_right_boundary 0.5 \
    --num_masked_solution_traces 1 \
    --rc_mode maj \
    --rc_temperature 0 \
    --rc_n_completions 10 \
    --threshold 0.999 \
    --note default \
    --generator ours \
    --add_solution_trace