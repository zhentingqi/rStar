CUDA_VISIBLE_DEVICES=0 python run_src/do_generate.py \
    --dataset_name AMC2024 \
    --test_json_filename test_all \
    --model_ckpt meta-llama/Meta-Llama-3-8B-Instruct \
    --note amc_v2 \
    --num_rollouts 32 \
    --disable_a5