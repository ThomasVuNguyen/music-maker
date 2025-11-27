import torch
from datasets import load_dataset
import numpy as np
from encodec import EncodecModel
from encodec.utils import convert_audio
from tqdm import tqdm
import os

# --- Configuration ---
OUTPUT_FILE = "music_tokens.npy"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("Loading EnCodec model...")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)  # 6 kbps
    model = model.to(DEVICE)
    model.eval()
    print(f"EnCodec loaded on {DEVICE}")

    print("Loading dataset...")
    ds = load_dataset("lewtun/music_genres", split="train")
    
    all_tokens = []

    print("Tokenizing dataset...")
    with torch.no_grad():
        for i in tqdm(range(0, len(ds), BATCH_SIZE)):
            batch_end = min(i + BATCH_SIZE, len(ds))
            batch = ds[i:batch_end]
            
            # Process each audio in the batch
            batch_tokens = []
            for audio_data in batch['audio']:
                # Convert to torch tensor
                audio = torch.tensor(audio_data['array'], dtype=torch.float32)
                sr = audio_data['sampling_rate']
                
                # Ensure mono
                if audio.ndim == 1:
                    audio = audio.unsqueeze(0)
                elif audio.shape[0] > 1:
                    audio = torch.mean(audio, dim=0, keepdim=True)
                
                # Convert to model's sample rate and format
                audio = convert_audio(audio, sr, model.sample_rate, model.channels)
                audio = audio.unsqueeze(0).to(DEVICE)  # [1, C, T]
                
                # Encode
                encoded_frames = model.encode(audio)
                codes = encoded_frames[0][0]  # [B, n_q, T] -> [n_q, T]
                
                # Flatten the multi-codebook tokens: [n_q, T] -> [n_q * T]
                # This creates a single sequence of tokens
                codes_flat = codes.squeeze(0).cpu().numpy()  # [n_q, T]
                
                # Store as [n_q, T] to preserve structure
                batch_tokens.append(codes_flat)
            
            all_tokens.extend(batch_tokens)

    # Convert to numpy array
    # Each item is [n_q, T] where n_q=8 (number of codebooks)
    print(f"\nTokenized {len(all_tokens)} audio files")
    print(f"Token shape per file: {all_tokens[0].shape}")

    # Save as object array to handle variable-length sequences
    all_tokens_array = np.empty(len(all_tokens), dtype=object)
    for i, tokens in enumerate(all_tokens):
        all_tokens_array[i] = tokens

    np.save(OUTPUT_FILE, all_tokens_array, allow_pickle=True)
    print(f"Tokens saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
