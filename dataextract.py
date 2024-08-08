from datasets import load_dataset

# Load the dataset using the custom script
dataset = load_dataset('/Users/jayeshgajbhar/LLM-Forge/dataextract.py', split='train')

# Iterate over the dataset to get the text content
for example in dataset:
    text_content = example['text']
    # Process the text content as needed
    print(text_content)  # or save to a file