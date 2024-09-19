CUDA_VISIBLE_DEVICES=0 python run_src/do_generate.py \
    --dataset_name AMC \
    --test_json_filename test_all \
    --model_ckpt ../Mistral-7B-v0.1 \
    --note default \
    --num_rollouts 16
