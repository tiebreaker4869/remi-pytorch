import os
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, wasserstein_distance
from typing import List, Dict
import argparse
from tqdm import tqdm


def extract_features(pm: pretty_midi.PrettyMIDI) -> Dict[str, float]:
    notes = [note for instr in pm.instruments for note in instr.notes if not instr.is_drum]
    if not notes:
        return None

    pitches = [note.pitch for note in notes]
    pitch_count = len(set(pitches))
    pitch_range = max(pitches) - min(pitches)
    pitch_intervals = [abs(pitches[i] - pitches[i - 1]) for i in range(1, len(pitches))]
    avg_pitch_interval = np.mean(pitch_intervals) if pitch_intervals else 0

    onset_times = sorted([note.start for note in notes])
    ioi = [onset_times[i] - onset_times[i - 1] for i in range(1, len(onset_times))]
    avg_ioi = np.mean(ioi) if ioi else 0

    start_time = min([note.start for note in notes])
    end_time = max([note.end for note in notes])
    duration = end_time - start_time
    duration = duration if duration > 0 else 1  # avoid div by zero

    note_density = len(notes) / duration
    pitch_density = pitch_count / len(notes) if len(notes) > 0 else 0

    return {
        "pitch_density": pitch_density,
        "pitch_range": pitch_range,
        "avg_pitch_interval": avg_pitch_interval,
        "note_density": note_density,
        "avg_ioi": avg_ioi
    }


def extract_features_from_dir(directory: str) -> Dict[str, List[float]]:
    all_feats = {k: [] for k in ["pitch_density", "pitch_range", "avg_pitch_interval", "note_density", "avg_ioi"]}
    midi_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith((".mid", ".midi"))]
    for fpath in tqdm(midi_files, desc=f"Processing {directory}"):
        try:
            pm = pretty_midi.PrettyMIDI(fpath)
            feats = extract_features(pm)
            if feats:
                for k in all_feats:
                    all_feats[k].append(feats[k])
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
    return all_feats


def kl_divergence(p, q, epsilon=1e-10):
    p = np.asarray(p) + epsilon
    q = np.asarray(q) + epsilon
    p /= p.sum()
    q /= q.sum()
    return entropy(p, q)

def compare_and_plot_features(gen_feats: Dict[str, List[float]], ref_feats: Dict[str, List[float]], output_dir="./plots"):
    os.makedirs(output_dir, exist_ok=True)
    for metric in gen_feats.keys():
        gen_vals = np.array(gen_feats[metric])
        ref_vals = np.array(ref_feats[metric])
        gen_hist, bins = np.histogram(gen_vals, bins=30, density=True)
        ref_hist, _ = np.histogram(ref_vals, bins=bins, density=True)
        kl = kl_divergence(gen_hist, ref_hist)
        wd = wasserstein_distance(gen_vals, ref_vals)
        print(f"[{metric}] KL Divergence: {kl:.4f}, Wasserstein Distance: {wd:.4f}")
        print(f"[{metric}] Mean (gen): {gen_vals.mean():.3f}, Mean (ref): {ref_vals.mean():.3f}")
        plt.figure(figsize=(6, 4))
        sns.kdeplot(ref_vals, label="Reference", fill=True)
        sns.kdeplot(gen_vals, label="Generated", fill=True)
        plt.title(f"{metric} (KL={kl:.3f}, WD={wd:.3f})")
        plt.xlabel(metric)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=str, required=True, help="Directory of generated MIDI files")
    parser.add_argument("--reference", type=str, required=True, help="Directory of reference MIDI files")
    parser.add_argument("--output", type=str, default="./plots", help="Output directory for plots")
    args = parser.parse_args()

    gen_feats = extract_features_from_dir(args.generated)
    ref_feats = extract_features_from_dir(args.reference)
    compare_and_plot_features(gen_feats, ref_feats, args.output)


if __name__ == "__main__":
    main()

