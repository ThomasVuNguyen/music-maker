from datasets import load_dataset
import torchaudio

def inspect():
    print("Loading dataset...")
    ds = load_dataset("lewtun/music_genres", split="train")
    print(f"Dataset size: {len(ds)}")
    
    sample = ds[0]
    print("Sample keys:", sample.keys())
    
    audio = sample['audio']
    print("Audio info:", audio)
    
    # Check sample rate and shape
    array = audio['array']
    sr = audio['sampling_rate']
    print(f"Array shape: {array.shape}, Sampling rate: {sr}")
    print(f"Duration: {array.shape[0] / sr:.2f}s")

if __name__ == "__main__":
    inspect()
