import os
import shutil

source_dir = "data/Pop1K7/midi_analyzed"
target_dir = "data/POP1K7"

os.makedirs(target_dir, exist_ok=True)

existing_filenames = set()

for root, _, files in os.walk(source_dir):
    for file in files:
        if 'v' in file.lower():
            continue
        if file.lower().endswith(".mid") or file.lower().endswith(".midi"):
            original_path = os.path.join(root, file)
            base_name = os.path.basename(file)
            new_path = os.path.join(target_dir, base_name)

            counter = 1
            while new_path in existing_filenames or os.path.exists(new_path):
                name, ext = os.path.splitext(base_name)
                new_base = f"{name}_{counter}{ext}"
                new_path = os.path.join(target_dir, new_base)
                counter += 1

            shutil.copy2(original_path, new_path)
            existing_filenames.add(new_path)
