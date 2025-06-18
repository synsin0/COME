# logger cfg
work_dir = None
extra_tag = 'default'
log_interval = 5

# randomness cfg
seed=42
deterministic = False
diff_rank_seed = False

# evaluation cfg
val_interval = 1
save_results_after_eval = False
comparison_mode = 'MIoU:+'

# checkpoint saving cfg
save_ckpt_interval = 1
max_ckpt_save_num = 5
save_best_ckpt = True

# resume cfg
resume = 'auto'
resume_status = True
load_from = None

stage_one_config = None
stage_one_ckpt = None