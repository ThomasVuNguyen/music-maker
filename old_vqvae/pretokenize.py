import torch
from datasets import load_dataset
import numpy as np
from train_tokenizer import VQVAE, NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST, DEVICE, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- Configuration ---
TOKENIZER_CHECKPOINT = "tokenizer_checkpoint.pt"
OUTPUT_FILE = "music_tokens.npy"
BATCH_SIZE = 8

def main():
    print("Loading tokenizer...")
    model = VQVAE(NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST).to(DEVICE)
    
    if os.path.exists(TOKENIZER_CHECKPOINT):
        checkpoint = torch.load(TOKENIZER_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded tokenizer checkpoint.")
    else:
        print("Warning: Tokenizer checkpoint not found. Using random weights (for testing only).")

    model.eval()

    print("Loading dataset...")
    ds = load_dataset("lewtun/music_genres", split="train")
    dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

    all_tokens = []

    print("Pre-tokenizing dataset...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(DEVICE)
            
            # Encode
            z = model.encoder(batch)
            
            # Quantize to get indices
            # VQ-VAE forward does: loss, quantized, perplexity, encoding_indices
            # But we need indices directly. Let's look at VectorQuantizer.forward
            # It returns: loss, quantized, perplexity, encoding_indices
            
            # We can call model.vq(z)
            _, _, _, encoding_indices = model.vq(z)
            
            # encoding_indices is [B*T, 1] -> reshape to [B, T]
            # z shape is [B, C, T_latent]
            # vq expects [B, C, T] -> permutes to [B, T, C] -> flattens to [B*T, C]
            
            # The encoding_indices returned by VectorQuantizer is [B*T, 1]
            # We need to reshape it back to [B, T_latent]
            
            B, C, T = z.shape
            indices = encoding_indices.view(B, T)
            
            all_tokens.append(indices.cpu().numpy().astype(np.uint16))

    all_tokens = np.concatenate(all_tokens, axis=0)
    print(f"Saved {all_tokens.shape[0]} sequences of length {all_tokens.shape[1]}")
    np.save(OUTPUT_FILE, all_tokens)
    print(f"Tokens saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
