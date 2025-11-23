import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import torchaudio
import os
from tqdm import tqdm

# --- Configuration ---
SAMPLE_RATE = 44100
BATCH_SIZE = 4 # Reduced to avoid OOM
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 64
NUM_EMBEDDINGS = 512
COMMITMENT_COST = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "tokenizer_checkpoint.pt"

print(f"Using device: {DEVICE}")

# --- Model Definition ---

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

    def forward(self, inputs):
        # inputs: [B, C, T] -> [B, T, C]
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self._embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # [B, T, C] -> [B, C, T]
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encoding_indices

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super(Encoder, self).__init__()
        # Strides: 4, 4, 4, 5 = 320x downsampling
        # 1323000 / 320 = 4134.375 -> ~4135 tokens
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, embedding_dim, 5, stride=5, padding=0)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_channels, out_channels):
        super(Decoder, self).__init__()
        # Inverse of Encoder
        self.net = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, hidden_channels, 5, stride=5, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels, hidden_channels, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels, hidden_channels, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels, out_channels, 4, stride=4, padding=0)
        )

    def forward(self, x):
        return self.net(x)

class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(1, 128, embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, 128, 1)

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity

# --- Data Loading ---

def collate_fn(batch):
    # Pad or trim to fixed length if necessary, but dataset seems consistent 30s
    # We'll just stack them.
    # batch is list of dicts
    tensors = []
    target_length = 1323200 # Divisible by 320 (320 * 4135)
    
    for item in batch:
        # item['audio']['array'] is numpy array
        audio = torch.tensor(item['audio']['array'], dtype=torch.float32)
        # Ensure [1, T]
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            
        # Pad or crop
        current_length = audio.shape[1]
        if current_length > target_length:
            audio = audio[:, :target_length]
        elif current_length < target_length:
            padding = target_length - current_length
            audio = torch.nn.functional.pad(audio, (0, padding))
            
        tensors.append(audio)
    
    # Stack
    return torch.stack(tensors)

def main():
    print("Loading dataset...")
    ds = load_dataset("lewtun/music_genres", split="train")
    # Use a subset for quick testing if needed, but goal is all data.
    # ds = ds.select(range(100)) 
    
    dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    model = VQVAE(NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in pbar:
            batch = batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                vq_loss, recon_batch, perplexity = model(batch)
                
                recon_loss = torch.mean((recon_batch - batch)**2)
                loss = recon_loss + vq_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            
            pbar.set_postfix({
                "Loss": loss.item(),
                "Recon": recon_loss.item(),
                "VQ": vq_loss.item(),
                "Ppl": perplexity.item()
            })
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, SAVE_PATH)
        print(f"Saved checkpoint to {SAVE_PATH}")

if __name__ == "__main__":
    main()
