import torch
import torch.nn as nn
import torchaudio
import numpy as np
import os
import math
from tqdm import tqdm
from encodec import EncodecModel
from encodec.utils import convert_audio

# Import model from updated train_autoregressive
from train_autoregressive import MultiCodebookGPT, VOCAB_SIZE, NUM_CODEBOOKS, EMBED_DIM, NUM_LAYERS, NUM_HEADS, SEQ_LEN, DROPOUT

# Configuration
DEVICE = "cpu"  # Use CPU to avoid OOM during generation

def load_models(device):
    print("Loading EnCodec...")
    encodec = EncodecModel.encodec_model_24khz()
    encodec.set_target_bandwidth(6.0)
    encodec = encodec.to(device)
    encodec.eval()
    print("EnCodec loaded.")

    print("Loading GPT...")
    gpt = MultiCodebookGPT(VOCAB_SIZE, NUM_CODEBOOKS, EMBED_DIM, NUM_LAYERS, NUM_HEADS, 
                           max_len=SEQ_LEN, dropout=DROPOUT).to(device)
    if os.path.exists("autocompletion_model.pt"):
        gpt.load_state_dict(torch.load("autocompletion_model.pt", map_location=device))
        print("GPT loaded.")
    else:
        print("Warning: autocompletion_model.pt not found. Using random weights.")

    gpt.eval()
    return encodec, gpt

def generate(input_path, output_path, duration_seconds, device):
    encodec, gpt = load_models(device)
    
    print(f"Processing {input_path}...")
    wav, sr = torchaudio.load(input_path)
    
    # Convert to EnCodec format
    wav = convert_audio(wav, sr, encodec.sample_rate, encodec.channels)
    wav = wav.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Encode input audio
        print("Encoding input...")
        encoded_frames = encodec.encode(wav)
        codes = encoded_frames[0][0]  # [1, n_q, T]
        
        print(f"Encoded tokens shape: {codes.shape}")
        
        # Use full input as prompt
        prompt = codes  # [1, n_q, T]
        prompt_len = prompt.shape[2]
        
        print(f"Prompting with {prompt_len} tokens (approx {prompt_len * 320 / encodec.sample_rate:.2f}s)...")
        
        # Calculate tokens needed for target duration
        # EnCodec at 24kHz produces ~75 tokens per second
        tokens_per_second = encodec.sample_rate / 320
        target_tokens = int(duration_seconds * tokens_per_second)
        max_new_tokens = target_tokens - prompt_len
        
        if max_new_tokens <= 0:
            print("Warning: Input is longer than target duration. No new tokens generated.")
            max_new_tokens = 0
            
        print(f"Generating {max_new_tokens} new tokens...")
        
        # Generation loop
        generated = prompt
        
        for _ in tqdm(range(max_new_tokens), desc="Generating"):
            # Crop context if needed
            idx_cond = generated[:, :, -SEQ_LEN:] if generated.shape[2] > SEQ_LEN else generated
            
            # Forward pass
            logits, _ = gpt(idx_cond)  # [B, n_q, T, vocab_size]
            
            # Get logits for the last position, all codebooks
            next_token_logits = logits[:, :, -1, :]  # [B, n_q, vocab_size]
            
            # Sample from each codebook
            next_tokens = []
            for cb in range(NUM_CODEBOOKS):
                cb_logits = next_token_logits[:, cb, :] / 1.0  # Temperature
                probs = torch.nn.functional.softmax(cb_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
                next_tokens.append(next_token)
            
            next_tokens = torch.stack(next_tokens, dim=1)  # [B, n_q, 1]
            generated = torch.cat((generated, next_tokens), dim=2)
        
        print(f"Generated sequence length: {generated.shape[2]}")
        
        # Decode with EnCodec
        print("Decoding...")
        # Reconstruct encoded_frames format
        encoded_frames_out = [(generated, None)]
        out_audio = encodec.decode(encoded_frames_out)
        
    # Save
    print(f"Saving to {output_path}...")
    torchaudio.save(output_path, out_audio.squeeze(0).cpu(), encodec.sample_rate)
    print("Done.")

if __name__ == "__main__":
    INPUT_FILE = "5s.wav"
    OUTPUT_FILE = "generated.wav"
    DURATION = 30.0
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
    else:
        generate(INPUT_FILE, OUTPUT_FILE, DURATION, DEVICE)
