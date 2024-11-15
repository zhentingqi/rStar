CUDA_VISIBLE_DEVICES=0 python run_src/do_generate.py \
    --dataset_name AMC2024 \
    --test_json_filename test_all \
    --model_ckpt meta-llama/Llama-3.1-8B \
    --note amc \
    --num_rollouts 16