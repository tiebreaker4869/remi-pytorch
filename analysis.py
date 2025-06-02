import os
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    midi_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith((".mid", ".midi"))]
    for fpath in tqdm(midi_files, desc=f"Processing {os.path.basename(directory)}"):
        try:
            pm = pretty_midi.PrettyMIDI(fpath)
            feats = extract_features(pm)
            if feats:
                for k in all_feats:
                    all_feats[k].append(feats[k])
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
    return all_feats

def plot_and_save_stats(feats: Dict[str, List[float]], output_dir="./stats_output"):
    os.makedirs(output_dir, exist_ok=True)
    summary_lines = []
    for metric, values in feats.items():
        values = np.array(values)
        summary_lines.append(f"{metric}: Mean={values.mean():.4f}, Median={np.median(values):.4f}, Min={values.min():.4f}, Max={values.max():.4f}, Std={values.std():.4f}, N={len(values)}")
        plt.figure(figsize=(6, 4))
        sns.histplot(values, bins=30, kde=True, color='dodgerblue')
        plt.title(f"{metric} distribution")
        plt.xlabel(metric)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}.png"))
        plt.close()

    with open(os.path.join(output_dir, "feature_summary.txt"), "w") as f:
        for line in summary_lines:
            print(line)
            f.write(line + "\n")
    print(f"Summary and plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", type=str, required=True, help="Directory of MIDI files")
    parser.add_argument("--output", type=str, default="./stats_output", help="Output directory for plots and summary")
    args = parser.parse_args()

    feats = extract_features_from_dir(args.midi_dir)
    plot_and_save_stats(feats, args.output)

if __name__ == "__main__":
    main()
