CUDA_VISIBLE_DEVICES=0 python run_src/do_generate.py \
    --dataset_name AMC2024 \
    --test_json_filename test_all \
    --model_ckpt microsoft/Phi-3-mini-4k-instruct \
    --note amc \
    --num_rollouts 16 \
    --verbose  
