CUDA_VISIBLE_DEVICES=2 python run_src/do_generate.py \
    --dataset_name AMC \
    --test_json_filename test_all \
    --model_ckpt ../Phi-3-mini-4k-instruct_folio \
    --note default \
    --num_rollouts 32 \
    --run_outputs_root /mnt/teamdrive/jiahang/rStar/ \
    --eval_outputs_root /mnt/teamdrive/jiahang/rStar/
