import os
from midi2audio import FluidSynth
from tqdm import tqdm
import argparse

def convert_midi_dir_to_wav(input_dir, output_dir, soundfont=None, sample_rate=44100):
    os.makedirs(output_dir, exist_ok=True)
    if soundfont is None:
        soundfont = "./resources/GeneralUser-GS/GeneralUser-GS.sf2"
        print(f"[Info] Using default soundfont: {soundfont}")

    fs = FluidSynth(sound_font=soundfont, sample_rate=sample_rate)

    midi_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".mid", ".midi"))])
    for midi_file in tqdm(midi_files, desc="Converting MIDI to WAV"):
        midi_path = os.path.join(input_dir, midi_file)
        wav_path = os.path.join(output_dir, os.path.splitext(midi_file)[0] + ".wav")
        try:
            fs.midi_to_audio(midi_path, wav_path)
        except Exception as e:
            print(f"[Error] Failed to convert {midi_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input directory with MIDI files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for WAV files")
    parser.add_argument("--soundfont", type=str, default=None, help="Path to SoundFont .sf2 file")
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate for output WAV files")
    args = parser.parse_args()

    convert_midi_dir_to_wav(args.input, args.output, soundfont=args.soundfont, sample_rate=args.sr)
