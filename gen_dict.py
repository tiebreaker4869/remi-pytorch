from collections import Counter
import pickle
import glob
import utils
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

all_elements= []
for midi_file in glob.glob("./data/POP909/*.mid*", recursive=True):
    events = extract_events(midi_file) # If you're analyzing chords, use `extract_events(midi_file, chord=True)`
    for event in events:
        element = '{}_{}'.format(event.name, event.value)
        all_elements.append(element)

counts = Counter(all_elements)
event2word = {c: i for i, c in enumerate(counts.keys())}
word2event = {i: c for i, c in enumerate(counts.keys())}
pickle.dump((event2word, word2event), open('dictionary/pop909_dictionary.pkl', 'wb'))