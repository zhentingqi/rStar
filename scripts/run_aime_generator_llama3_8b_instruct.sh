CUDA_VISIBLE_DEVICES=0 python run_src/do_generate.py \
    --dataset_name AIME \
    --test_json_filename test_all \
    --model_ckpt ../Meta-Llama-3-8B-Instruct  \
    --note v0_v0_v0 \
    --num_rollouts 32 \
    --run_outputs_root /mnt/teamdrive/jiahang/rStar/ \
    --eval_outputs_root /mnt/teamdrive/jiahang/rStar/
