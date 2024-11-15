CUDA_VISIBLE_DEVICES=0 python run_src/do_generate.py \
    --dataset_name GSM8K \
    --test_json_filename test_all \
    --model_ckpt microsoft/Phi-3-mini-4k-instruct \
    --note default \
    --num_rollouts 16
