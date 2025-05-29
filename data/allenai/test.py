import tiktoken
import numpy as np
import os

enc = tiktoken.get_encoding("gpt2")
ignore_index = -100

def inspect_sample_data(split='train', num_samples=0):
    x_file = f'{split}_x.bin'
    y_file = f'{split}_y.bin'
    
    # Load the memory-mapped arrays
    arr_x = np.memmap(x_file, dtype=np.uint16, mode='r')
    arr_y = np.memmap(y_file, dtype=np.int64, mode='r')
    
    print(f"\nInspecting {split} split:")
    print(f"Total tokens in X: {len(arr_x):,}")
    print(f"Total tokens in Y: {len(arr_y):,}")
    
    # Sample a few random positions
    sample_positions = np.random.choice(len(arr_x), size=num_samples, replace=False)
    
    for pos in sample_positions[:num_samples]:
        # Look at ~100 tokens around this position
        start_idx = max(0, pos - 50)
        end_idx = min(len(arr_x), pos + 50)
        
        x_sample = arr_x[start_idx:end_idx]
        y_sample = arr_y[start_idx:end_idx]
        
        print(f"\n--- Sample at position {pos} ---")
        print("X tokens (first 20):", x_sample[:20].tolist())
        print("Y tokens (first 20):", y_sample[:20].tolist())
        
        # Decode to see actual text
        try:
            decoded_x = enc.decode(x_sample.tolist())
            print("Decoded X:", repr(decoded_x[:200]))
        except:
            print("Could not decode X sample")
        
        # Check masking pattern
        masked_positions = np.where(y_sample == ignore_index)[0]
        non_masked_positions = np.where(y_sample != ignore_index)[0]
        
        print(f"Masked positions: {len(masked_positions)}/{len(y_sample)}")
        print(f"Non-masked positions: {len(non_masked_positions)}/{len(y_sample)}")

inspect_sample_data('train')

def check_conversation_structure():
    """Look for conversation markers in the tokenized data"""
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode the prefixes to look for them in data
    user_tokens = enc.encode_ordinary("USER:\n")
    assistant_tokens = enc.encode_ordinary("ASSISTANT:\n")
    system_tokens = enc.encode_ordinary("Below is a conversation between a user and an assitant")
    
    print("Expected token patterns:")
    print(f"USER prefix: {user_tokens}")
    print(f"ASSISTANT prefix: {assistant_tokens}")
    print(f"System prompt start: {system_tokens[:10]}")
    
    # Check if these patterns appear in your data
    arr_x = np.memmap('train_x.bin', dtype=np.uint16, mode='r')
    
    # Look for system prompt at the beginning
    first_tokens = arr_x[:50].tolist()
    print(f"\nFirst 50 tokens: {first_tokens}")
    
    # Search for conversation markers
    x_list = arr_x[:10000].tolist()  # Check first 10k tokens
    
    user_pattern_found = any(x_list[i:i+len(user_tokens)] == user_tokens 
                           for i in range(len(x_list)-len(user_tokens)))
    assistant_pattern_found = any(x_list[i:i+len(assistant_tokens)] == assistant_tokens 
                                for i in range(len(x_list)-len(assistant_tokens)))
    
    print(f"USER pattern found: {user_pattern_found}")
    print(f"ASSISTANT pattern found: {assistant_pattern_found}")

check_conversation_structure()

def validate_masking_pattern():
    """Check that masking follows the expected pattern"""
    enc = tiktoken.get_encoding("gpt2")
    
    arr_x = np.memmap('train_x.bin', dtype=np.uint16, mode='r')
    arr_y = np.memmap('train_y.bin', dtype=np.int64, mode='r')
    
    # Look at first 1000 tokens
    sample_size = 1000
    x_sample = arr_x[:sample_size]
    y_sample = arr_y[:sample_size]
    
    assistant_tokens = enc.encode_ordinary("ASSISTANT:\n")
    
    print("Checking masking pattern...")
    
    # Find where ASSISTANT tokens appear
    for i in range(len(x_sample) - len(assistant_tokens)):
        if x_sample[i:i+len(assistant_tokens)].tolist() == assistant_tokens:
            print(f"\nFound ASSISTANT at position {i}")
            
            # Check what's masked vs unmasked around this position
            context_start = max(0, i-10)
            context_end = min(len(x_sample), i+20)
            
            x_context = x_sample[context_start:context_end]
            y_context = y_sample[context_start:context_end]
            
            print("X context:", x_context.tolist())
            print("Y context:", y_context.tolist())
            print("Masked positions:", (y_context == ignore_index).tolist())
            
            break

validate_masking_pattern()

def check_statistics():
    """Basic statistical checks"""
    for split in ['train', 'val']:
        if os.path.exists(f'{split}_x.bin'):
            arr_x = np.memmap(f'{split}_x.bin', dtype=np.uint16, mode='r')
            arr_y = np.memmap(f'{split}_y.bin', dtype=np.int64, mode='r')
            
            print(f"\n{split.upper()} STATISTICS:")
            print(f"Total tokens: {len(arr_x):,}")
            
            # Token distribution
            unique_x, counts_x = np.unique(arr_x, return_counts=True)
            print(f"Unique tokens in X: {len(unique_x)}")
            print(f"Most common tokens: {unique_x[np.argsort(counts_x)[-5:]]}")
            
            # Masking statistics
            masked_count = np.sum(arr_y == ignore_index)
            unmasked_count = len(arr_y) - masked_count
            
            print(f"Masked tokens: {masked_count:,} ({masked_count/len(arr_y)*100:.1f}%)")
            print(f"Unmasked tokens: {unmasked_count:,} ({unmasked_count/len(arr_y)*100:.1f}%)")

check_statistics()