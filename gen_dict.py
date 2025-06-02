import argparse
from collections import Counter
import pickle
import glob
import utils
from tqdm import tqdm

def extract_events(input_path, chord=True):
    note_items, tempo_items = utils.read_items(input_path)
    note_items = utils.quantize_items(note_items)
    max_time = note_items[-1].end
    if chord:
        chord_items = utils.extract_chords(note_items)
        items = chord_items + tempo_items + note_items
    else:
        items = tempo_items + note_items
    groups = utils.group_items(items, max_time)
    events = utils.item2event(groups)
    return events

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing MIDI files')
    parser.add_argument('--output', type=str, required=True, help='Output path for dictionary pickle')
    parser.add_argument('--with_chord', action='store_true', help='Whether to extract and include chord events')
    args = parser.parse_args()

    all_elements = []
    pattern = args.data_dir.rstrip('/') + '/*.mid*'
    midi_files = glob.glob(pattern, recursive=True)
    for midi_file in tqdm(midi_files, desc="Processing MIDI files"):
        events = extract_events(midi_file, chord=args.with_chord)
        for event in events:
            element = '{}_{}'.format(event.name, event.value)
            all_elements.append(element)

    counts = Counter(all_elements)
    event2word = {c: i for i, c in enumerate(counts.keys())}
    word2event = {i: c for i, c in enumerate(counts.keys())}
    pickle.dump((event2word, word2event), open(args.output, 'wb'))

if __name__ == "__main__":
    main()
