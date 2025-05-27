import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# --- Configuration ---
num_proc = 6
num_proc_load_dataset = num_proc
IGNORE_INDEX = -100  # Standard ignore index for CrossEntropyLoss
enc = tiktoken.get_encoding("gpt2")
# ---

def process_example_for_sft(example):
    prompt_str, response_str = example['text'], example['output']

    prompt_ids = enc.encode_ordinary(prompt_str)
    response_ids = enc.encode_ordinary(response_str)

    # x_ids: Tokens fed to the model (prompt + response)
    x_ids = prompt_ids + response_ids
    
    # y_ids: Target tokens (shifted version of prompt + response + EOT)
    # These are the tokens the model should predict at each step.
    # For input P1 P2 R1 R2, targets are P2 R1 R2 EOT.
    y_ids_full_sequence = prompt_ids + response_ids + [enc.eot_token]
    y_ids = y_ids_full_sequence[1:] # Shifted target

    # Mask y_ids: tokens corresponding to the prompt part should be IGNORE_INDEX
    # The first `len(prompt_ids)` tokens in y_ids are predictions generated *from* prompt tokens.
    y_ids_masked = list(y_ids) # Make mutable for masking
    
    # Number of positions in y_ids to mask is len(prompt_ids)
    # (e.g., if prompt_ids = [P1, P2, P3], then first 3 elements of y_ids are masked)
    num_prompt_tokens_to_mask_in_y = len(prompt_ids)

    for i in range(min(num_prompt_tokens_to_mask_in_y, len(y_ids_masked))):
        y_ids_masked[i] = IGNORE_INDEX
        
    return {'x_ids': x_ids, 'y_ids_masked': y_ids_masked, 'len_x': len(x_ids), 'len_y': len(y_ids_masked)}

if __name__ == '__main__':
    dataset_hf = load_dataset("tatsu-lab/alpaca", num_proc=num_proc_load_dataset)

    split_dataset = dataset_hf["train"].train_test_split(test_size=0.05, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    # Tokenize the dataset using the new SFT processing function
    tokenized_sft = split_dataset.map(
        process_example_for_sft,
        remove_columns=['instruction', 'input', 'output', 'text'], # Original columns
        desc="tokenizing the splits for SFT",
        num_proc=num_proc,
    )

    for split, dset in tokenized_sft.items():
        # Calculate total lengths for x and y streams
        arr_len_x = np.sum(dset['len_x'], dtype=np.uint64)
        arr_len_y = np.sum(dset['len_y'], dtype=np.uint64)
        # assert arr_len_x == arr_len_y, f"Mismatch in total X and Y lengths for split {split}"

        filename_x = os.path.join(os.path.dirname(__file__), f'{split}_x.bin')
        filename_y = os.path.join(os.path.dirname(__file__), f'{split}_y.bin')
        
        dtype_x = np.uint16 # For token IDs (0-50256)
        dtype_y = np.int64  # For target IDs (can be -100 or 0-50256)

        arr_x = np.memmap(filename_x, dtype=dtype_x, mode='w+', shape=(arr_len_x,))
        arr_y = np.memmap(filename_y, dtype=dtype_y, mode='w+', shape=(arr_len_y,))
        
        total_batches = 1024 # As in original script, for sharded writing
        idx_x, idx_y = 0, 0

        for batch_idx in tqdm(range(total_batches), desc=f'writing {split} x/y bins'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            
            arr_batch_x = np.concatenate(batch['x_ids'])
            arr_batch_y = np.concatenate(batch['y_ids_masked'])
            
            arr_x[idx_x : idx_x + len(arr_batch_x)] = arr_batch_x
            arr_y[idx_y : idx_y + len(arr_batch_y)] = arr_batch_y
            
            idx_x += len(arr_batch_x)
            idx_y += len(arr_batch_y)
            
        arr_x.flush()
        arr_y.flush()