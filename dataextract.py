import os
import lzma
from tqdm import tqdm


def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files


folder_path = '/Users/jayeshgajbhar/LLM-Forge/openwebtext'
output_file_train = "output_train_{}.txt"
output_file_val = "output_val_{}.txt"
vocab_file = "vocab.txt"
split_files = int(input("How many files would you like to split this into?"))

files = xz_files_in_dir(folder_path)
total_files = len(files)

split_index = int(total_files * 0.9)  # 90% for training
files_train = files[:split_index]
files_val = files[split_index:]

vocab = set()

# Helper function to process files and write output


def process_files(files, output_file_pattern):
    max_count = len(files) // split_files + 1
    for i in range(split_files):
        with open(output_file_pattern.format(i), "w", encoding="utf-8") as outfile:
            for count, filename in enumerate(tqdm(files[:max_count], total=max_count)):
                if count >= max_count:
                    break
                file_path = os.path.join(folder_path, filename)
                with lzma.open(file_path, "rt", encoding="utf-8") as infile:
                    text = infile.read()
                    outfile.write(text)
                    characters = set(text)
                    vocab.update(characters)
            files[:] = files[max_count:]


# Process training files
process_files(files_train, output_file_train)

# Process validation files
process_files(files_val, output_file_val)

# Write vocabulary to file
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in sorted(vocab):
        vfile.write(char + '\n')
