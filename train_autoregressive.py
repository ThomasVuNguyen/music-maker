import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import math

# --- Configuration ---
TOKENS_FILE = "music_tokens.npy"
BATCH_SIZE = 1 # Adjust based on VRAM
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
SEQ_LEN = 4096 # Context length (must be <= token sequence length)
VOCAB_SIZE = 512 # Matches NUM_EMBEDDINGS in tokenizer
EMBED_DIM = 768
NUM_LAYERS = 12
NUM_HEADS = 12
DROPOUT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "autocompletion_model.pt"

# --- Model Definition (GPT-style) ---

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Flash Attention if available (PyTorch 2.0+)
        # output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.1 if self.training else 0)
        
        # Manual implementation for compatibility/clarity
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.sa = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_len, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Create position indices
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

# --- Data Loading ---

class TokenDataset(Dataset):
    def __init__(self, tokens_file, seq_len):
        self.data = np.load(tokens_file) # [N, L]
        self.seq_len = seq_len
        # We can train on random crops or full sequences. 
        # Since L ~ 4135 and seq_len ~ 4096, we can just take the first 4096 or random crop.
        # Let's do random crop for robustness.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx] # [L]
        # If sequence is shorter than seq_len, pad? 
        # Our tokenizer produces fixed length ~4135, so it should be fine.
        
        if len(seq) <= self.seq_len + 1:
             # Pad if necessary (unlikely with current settings)
             pad = np.zeros(self.seq_len + 1 - len(seq), dtype=seq.dtype)
             seq = np.concatenate([seq, pad])
        
        # Random crop
        max_start = len(seq) - self.seq_len - 1
        start = np.random.randint(0, max_start + 1)
        chunk = seq[start : start + self.seq_len + 1]
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def main():
    if not os.path.exists(TOKENS_FILE):
        print(f"Error: {TOKENS_FILE} not found. Run pretokenize.py first.")
        return

    print("Loading dataset...")
    dataset = TokenDataset(TOKENS_FILE, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print("Initializing model...")
    # Max len needs to cover the sequence length
    model = GPT(VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, max_len=SEQ_LEN, dropout=DROPOUT).to(DEVICE)
    
    # Check parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params/1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits, loss = model(x, targets=y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
        
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Saved checkpoint to {SAVE_PATH}")

if __name__ == "__main__":
    main()
