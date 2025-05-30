import os
import shutil

# 源目录（POP909 根目录）
source_dir = "data/POP909-Dataset"
# 目标目录（一级目录放置所有 MIDI 文件）
target_dir = "data/POP909"

# 创建目标目录
os.makedirs(target_dir, exist_ok=True)

# 记录已存在文件名防止覆盖
existing_filenames = set()

# 遍历 source_dir 所有子目录和文件
for root, _, files in os.walk(source_dir):
    for file in files:
        if 'v' in file.lower():
            continue
        if file.lower().endswith(".mid") or file.lower().endswith(".midi"):
            original_path = os.path.join(root, file)
            base_name = os.path.basename(file)
            new_path = os.path.join(target_dir, base_name)

            # 如果文件名冲突，自动添加编号
            counter = 1
            while new_path in existing_filenames or os.path.exists(new_path):
                name, ext = os.path.splitext(base_name)
                new_base = f"{name}_{counter}{ext}"
                new_path = os.path.join(target_dir, new_base)
                counter += 1

            # 复制文件
            shutil.copy2(original_path, new_path)
            existing_filenames.add(new_path)

print("✅ 所有 MIDI 文件已复制到：", target_dir)
