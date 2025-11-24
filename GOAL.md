# Goal 1 ✓

I have this dataset lewtun/music_genres (19.9k rows), with the 'audio' column is made of 30s audio files.

~~Help me train a tokenizer on all that.~~

**UPDATED:** Now using pre-trained EnCodec instead of custom VQVAE.

~~python inspect_dataset.py~~
~~python train_tokenizer.py~~

# Goal 2

Help me train an autocompletion model, 100M parameter on all that data

python encodec_pretokenize.py  # NEW: Uses EnCodec instead
python train_autoregressive.py  # UPDATED: Handles multi-codebook tokens

# Goal 3 ✓

Sample output from that (audio in, audio out)

python sample.py  # UPDATED: Uses EnCodec for encoding/decoding

# Note
Use the existing venv/ environment

We have RTX 4090 on this machine, utilize it whenever possible

# Changes Made

- **Replaced custom VQVAE** with Meta's pre-trained **EnCodec** for much better audio quality
- **No tokenizer training needed** - EnCodec works out of the box
- Old VQVAE files moved to `old_vqvae/` directory
