import string
import os

# Define allowed characters
allowed_chars = set(
    string.ascii_letters +  # a-z, A-Z
    string.digits +         # 0-9
    " .,;!?()[]{}<>-_+=*&^%$#@/|\\\"':`~\n\t"  # Special symbols including space, tab, and newline
)

def clean_text_and_extract_vocab(file_paths):
    """
    Cleans the text files and extracts a vocabulary of allowed characters.

    Parameters:
        file_paths (list): List of file paths to clean.

    Returns:
        vocab (set): A set of unique allowed characters found in the text files.
    """
    vocab = set()

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

            # Filter allowed characters
            filtered_text = ''.join(c for c in text if c in allowed_chars)

            # Update vocabulary set
            vocab.update(filtered_text)

            # Optionally, write cleaned text back to the file
            with open(file_path, 'w', encoding='utf-8') as fw:
                fw.write(filtered_text)

    return vocab

# List of file paths
file_paths = ["data/output_train_0.txt", "data/output_val_0.txt"]

# Extract vocabulary
vocab = clean_text_and_extract_vocab(file_paths)

# Sort the vocabulary for consistent order
sorted_vocab = sorted(vocab)

# Write the vocabulary to vocab.txt, each character on a new line
with open("data/vocab.txt", 'w', encoding='utf-8') as vocab_file:
    for char in sorted_vocab:
        # Replace the space and newline characters with readable representations
        if char == ' ':
            vocab_file.write("<space>\n")
        elif char == '\n':
            vocab_file.write("<newline>\n")
        elif char == '\t':
            vocab_file.write("<tab>\n")
        else:
            vocab_file.write(char + '\n')

print("Vocabulary extracted and saved to vocab.txt")
print("Extracted Characters:", sorted_vocab)