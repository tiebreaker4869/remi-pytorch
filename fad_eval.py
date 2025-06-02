import os
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from scipy.linalg import sqrtm
import argparse

from transformers import ClapModel, ClapProcessor

def get_audio_files(directory):
    return sorted([os.path.join(directory, f)
                  for f in os.listdir(directory)
                  if f.lower().endswith('.wav')])

def load_audio(path, target_sr=48000, max_sec=10):
    # Load and ensure mono, float32, resample to target_sr, pad/truncate to max_sec
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = audio.astype(np.float32)
    maxlen = target_sr * max_sec
    if len(audio) < maxlen:
        audio = np.pad(audio, (0, maxlen - len(audio)), 'constant')
    else:
        audio = audio[:maxlen]
    return audio

def extract_clap_embeddings(audio_paths, processor, model, device, bs=8):
    """Return np.array shape (N, embed_dim)"""
    embeddings = []
    for i in tqdm(range(0, len(audio_paths), bs), desc="Embedding"):
        batch_files = audio_paths[i:i+bs]
        batch_waveforms = [load_audio(path) for path in batch_files]
        inputs = processor(audios=batch_waveforms, return_tensors="pt", sampling_rate=48000, padding=True)
        with torch.no_grad():
            input_tensor = inputs["input_features"].to(device)
            output = model.get_audio_features(input_tensor)
            emb = output.cpu().numpy()
        embeddings.append(emb)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings

def compute_fad(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet distance for two Gaussians."""
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    # numerical stability
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=str, required=True, help="Directory of generated wav files")
    parser.add_argument("--reference", type=str, required=True, help="Directory of reference wav files")
    parser.add_argument("--bs", type=int, default=4, help="Batch size for embedding extraction")
    parser.add_argument("--device", type=str, default="cuda", help="Device for CLAP (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    # Load CLAP model and processor
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    model.eval()

    # Get file lists
    gen_files = get_audio_files(args.generated)
    ref_files = get_audio_files(args.reference)
    print(f"Generated files: {len(gen_files)}, Reference files: {len(ref_files)}")

    # Embedding extraction
    print("Extracting embeddings for generated set...")
    gen_emb = extract_clap_embeddings(gen_files, processor, model, device, bs=args.bs)
    print("Extracting embeddings for reference set...")
    ref_emb = extract_clap_embeddings(ref_files, processor, model, device, bs=args.bs)

    # Compute statistics
    mu_gen, sigma_gen = np.mean(gen_emb, axis=0), np.cov(gen_emb, rowvar=False)
    mu_ref, sigma_ref = np.mean(ref_emb, axis=0), np.cov(ref_emb, rowvar=False)

    # Compute FAD
    fad = compute_fad(mu_gen, sigma_gen, mu_ref, sigma_ref)
    print(f"FAD (CLAP embedding): {fad:.4f}")

if __name__ == "__main__":
    main()
