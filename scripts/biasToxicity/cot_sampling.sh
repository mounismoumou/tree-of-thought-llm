python run.py \
    --task biasToxicity \
    --task_start_index 0 \
    --task_end_index 10 \
    --naive_run \
    --prompt_sample cot \
    --n_generate_sample 10 \
    --temperature 1.0 \
    ${@}
