import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset
import sys

# --- Configuration ---
ignore_index = -100  # Standard ignore index for CrossEntropyLoss
num_proc = 6
num_proc_load_dataset = num_proc
enc = tiktoken.get_encoding("gpt2")

USER_PREFIX = "USER:\n"
ASSISTANT_PREFIX = "ASSISTANT:\n"
END_OF_TURN_SUFFIX = "\n"
SYSTEM_PROMPT_TEXT = "Below is a conversation between a user and an assitant. Complete the assistant's response.\n"
# ---

def process_example_for_sft_multi_turn(example):
    processed_turns = {'x_ids': [], 'y_ids_masked': [], 'len_x': [], 'len_y': []}
    system_prompt_tokens = enc.encode_ordinary(SYSTEM_PROMPT_TEXT)
    current_context_tokens = list(system_prompt_tokens) # Initialize with system prompt

    for i in range(len(example['messages'])):
        message = example['messages'][i]

        if message['role'] == 'user':
            user_tokens = enc.encode_ordinary(USER_PREFIX + message['content'] + END_OF_TURN_SUFFIX)
            current_context_tokens.extend(user_tokens)

        elif message['role'] == 'assistant':
            context_for_this_turn = list(current_context_tokens) # Includes system prompt + prior user turns
            assistant_tokens = enc.encode_ordinary(ASSISTANT_PREFIX + message['content'] + END_OF_TURN_SUFFIX)

            x_ids = context_for_this_turn + assistant_tokens
            # y_ids_full_sequence should also start from this full context
            y_ids_full_sequence = context_for_this_turn + assistant_tokens + [enc.eot_token] 
            y_ids = y_ids_full_sequence[1:] 

            y_ids_masked = list(y_ids)
            num_context_tokens_to_mask_in_y = len(context_for_this_turn) # This will correctly mask the system prompt and user parts

            for j in range(min(num_context_tokens_to_mask_in_y, len(y_ids_masked))):
                y_ids_masked[j] = ignore_index

            processed_turns['x_ids'].append(x_ids)
            processed_turns['y_ids_masked'].append(y_ids_masked)
            processed_turns['len_x'].append(len(x_ids))
            processed_turns['len_y'].append(len(y_ids_masked))
            
            current_context_tokens.extend(assistant_tokens)

    return processed_turns

if __name__ == '__main__':
    dataset_hf = load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225", num_proc=num_proc_load_dataset)

    split_dataset = dataset_hf["train"].train_test_split(test_size=0.05, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    tokenized_sft = split_dataset.map(
        process_example_for_sft_multi_turn,
        remove_columns=['id', 'messages', 'source'],
        desc="tokenizing the splits for SFT (multi-turn with markers)",
        num_proc=num_proc,
    )

    # Now flatten the results
    print("Flattening results...")
    for split_name in tokenized_sft.keys():
        print(f"Flattening {split_name}...")
        
        all_data = {'x_ids': [], 'y_ids_masked': [], 'len_x': [], 'len_y': []}
        for batch in tqdm(tokenized_sft[split_name].select(range(1000)), desc=f"Processing {split_name}"):
            # Each batch contains lists of training examples
            all_data['x_ids'].extend(batch['x_ids'])
            all_data['y_ids_masked'].extend(batch['y_ids_masked'])
            all_data['len_x'].extend(batch['len_x'])
            all_data['len_y'].extend(batch['len_y'])
        
        tokenized_sft[split_name] = Dataset.from_dict(all_data)
        print(f"{split_name}: {len(tokenized_sft[split_name])} training examples")

    for split, dset in tokenized_sft.items():
        arr_len_x = np.sum(dset['len_x'], dtype=np.uint64)
        arr_len_y = np.sum(dset['len_y'], dtype=np.uint64)
        assert arr_len_x == arr_len_y, f"Mismatch in total X and Y lengths for split {split}"

        filename_x = os.path.join(os.path.dirname(__file__), f'{split}_x.bin')
        filename_y = os.path.join(os.path.dirname(__file__), f'{split}_y.bin')
        
        dtype_x = np.uint16
        dtype_y = np.int64

        arr_x = np.memmap(filename_x, dtype=dtype_x, mode='w+', shape=(arr_len_x,))
        arr_y = np.memmap(filename_y, dtype=dtype_y, mode='w+', shape=(arr_len_y,))
        
        current_idx_x, current_idx_y = 0, 0
        for i in tqdm(range(len(dset)), desc=f'writing {split} x/y bins'):
            single_x_ids = dset[i]['x_ids']
            single_y_ids_masked = dset[i]['y_ids_masked']

            arr_x[current_idx_x : current_idx_x + len(single_x_ids)] = np.array(single_x_ids, dtype=dtype_x)
            arr_y[current_idx_y : current_idx_y + len(single_y_ids_masked)] = np.array(single_y_ids_masked, dtype=dtype_y)
            
            current_idx_x += len(single_x_ids)
            current_idx_y += len(single_y_ids_masked)
        
        arr_x.flush()
        arr_y.flush()