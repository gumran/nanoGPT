import time

out_dir = 'out-allenai'
eval_interval = 5
eval_iters = 40
wandb_log = True # feel free to turn on
wandb_project = 'allenai'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'allenai'
init_from = 'gpt2-xl' # this is the largest GPT-2 model

instruction_tuning = True  # Enable instruction tuning for Alpaca dataset

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

max_iters = 200