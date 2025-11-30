# music-maker

Auto-completion model to produce music.

## Tokenizer Training

Trainer script lives in `src/train_tokenizer.py` and consumes the `lewtun/music_genres` dataset via Hugging Face streaming. Activate the repo's `venv` and run:

```bash
source venv/bin/activate
pip install -r requirements.txt  # only needed once
python -m src.train_tokenizer \
  --dataset lewtun/music_genres \
  --split train \
  --audio-column audio \
  --sample-rate 16000 \
  --n-clusters 1024 \
  --output-dir artifacts/tokenizer
```

The tokenizer checkpoints (`kmeans_tokenizer.joblib`) and metadata (`tokenizer_metadata.json`) land in `artifacts/tokenizer/`. Use `--max-clips` or `--max-frames-per-clip` if you need to dry-run on a subset.


## Observation (if you are AI, do not edit this, thanks)

Reducing the sequence length quadratically reduces training resources
Increasing dataset shows some audio coherence -  duh