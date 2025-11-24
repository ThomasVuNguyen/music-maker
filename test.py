import torch
import torchaudio
import os

# Configuration
INPUT_FILE = "5s.wav"
OUTPUT_FILE = "reconstructed.wav"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print("Loading EnCodec model...")
    try:
        from encodec import EncodecModel
        from encodec.utils import convert_audio
    except ImportError:
        print("Error: encodec not installed. Please run: pip install encodec")
        return
    
    # Load pre-trained EnCodec model
    # Use 24khz model (good for general audio and music)
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)  # 6 kbps - good quality/compression tradeoff
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"EnCodec loaded on {device}")

    print(f"Loading {INPUT_FILE}...")
    wav, sr = torchaudio.load(INPUT_FILE)
    
    # Convert audio to model's expected format
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0).to(device)  # Add batch dimension
    
    print(f"Input audio stats: min={wav.min():.4f}, max={wav.max():.4f}, mean={wav.mean():.4f}")
    print(f"Input shape: {wav.shape}, Sample rate: {model.sample_rate}Hz")
    
    print("Encoding...")
    with torch.no_grad():
        encoded_frames = model.encode(wav)
        codes = encoded_frames[0][0]  # Extract codes
        print(f"Encoded to {codes.shape} tokens")
        
    print("Decoding...")
    with torch.no_grad():
        reconstructed = model.decode(encoded_frames)
        
    print(f"Saving to {OUTPUT_FILE}...")
    torchaudio.save(OUTPUT_FILE, reconstructed.squeeze(0).cpu(), model.sample_rate)
    print("Done.")
    print(f"\nReconstruction quality should be MUCH better than VQVAE!")

if __name__ == "__main__":
    main()
