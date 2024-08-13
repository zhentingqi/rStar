python run_src/do_discriminate.py \
    --model_ckpt /path/to/model \
    --root_dir ${BASE_ROOT}${ROOT_DIR} \
    --dataset_name ${DATASET_NAME} \
    --note default



    --rc_mode mid \
    --mask_left_boundary 0.2 \
    --mask_right_boundary 0.5 \
    --num_masked_solution_traces 4 \
    --rc_temperature 1 \
    --rc_n_completions 1 \
    --threshold 0.999 \