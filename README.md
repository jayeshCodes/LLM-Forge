# LLM-Forge

## Overview

**LLM-Forge** is a project aimed at building a Large Language Model (LLM) from scratch, using custom Transformer architectures. The project implements two primary models:

1. **Bigram Model** : A simple language model that predicts the next word based on the previous word.
2. **GPT Model** : A more complex generative pre-trained transformer model designed for generating coherent and contextually relevant text.

This project serves as an exploration into the fundamentals of language models and the intricacies of transformer architectures. The aim is to provide insights and tools for developing custom LLMs using Python and popular deep learning libraries.

### Table of Contents

1. [Overview of Language Models](#overview-of-language-models)
   - [Bigram Model](#bigram-model)
   - [GPT Model](#gpt-model)
2. [Research Papers](#research-papers)

## Overview of Language Models

### Bigram Model

- The Bigram Model is a foundational language model that operates by predicting the probability of a word based on the immediately preceding word.
- Unlike more complex models, the Bigram Model considers only the last word for its predictions, making it a simple yet effective approach for understanding the basic dynamics of language modeling.

**Key Concepts** :

- **Word Pairs**: The model learns relationships between consecutive words (bigrams) in a text corpus, where each bigram consists of two words that occur sequentially.

- **Probability Calculation**: The model estimates the probability of a word \(w_i\) following another word \(w_{i-1}\) using the formula:

  \[
   **P(w_i | w_{i-1}) = Count(w_{i-1}, w_i) / Count(w_{i-1})**
  \]

  Count(w_{i-1}, w_i) represents the number of times the word w_i follows the word w_{i-1} in the corpus.
  Count(w_{i-1}) is the number of times the word w_{i-1} appears in the corpus.

- **Markov Assumption**: The Bigram Model relies on the Markov assumption, which states that the probability of a word depends only on its immediate predecessor, not on any earlier words.

**Advantages** :
- **Simplicity**: Easy to implement and understand, making it suitable for introductory exploration of language models.
- **Efficiency**: Requires minimal computational resources due to its reliance on only the preceding word.
- **Baseline Performance**: Serves as a baseline for comparing more sophisticated models like n-gram models and transformers.

**Limitations** :
- Context Limitation: Fails to capture dependencies between non-adjacent words, resulting in a limited understanding of context.
- Data Sparsity: Struggles with rare word combinations that may not appear frequently in the training corpus.
- Vocabulary Sensitivity: The model’s performance heavily depends on the quality and size of the vocabulary.

**Use Cases** :
- Text Prediction: Simple applications like text autocompletion or predictive text systems.
- Language Analysis: Analyzing word pair frequencies and basic sentence structures in text data.

**Implementation** :

In this project, the Bigram Model is implemented as a baseline to demonstrate the progression from simple statistical models to more advanced neural network-based models like the GPT model (implemented in **bigram.ipynb**).

### GPT Model

- The GPT Model (Generative Pre-trained Transformer) is a deep learning model designed to generate coherent and contextually relevant text.
- It is built on the Transformer architecture, which leverages self-attention mechanisms to process input sequences in parallel, allowing the model to capture long-range dependencies in the data more effectively than traditional RNN-based models.

**Key Concepts** :

1. Transformer Architecture:
   - The Transformer architecture, introduced in the “Attention Is All You Need” paper, is the backbone of the GPT model. It replaces recurrent and convolutional layers with a purely attention-based mechanism, enabling the model to handle dependencies between words over long distances without the need for sequential data processing.

![image](https://github.com/user-attachments/assets/0f34fa22-7293-404c-a5dd-6d003a411489)

2. Self-Attention Mechanism:
- Self-attention allows the model to weigh the importance of different words in a sequence when encoding a particular word. In GPT, this mechanism helps the model understand the context by considering all the words in the input sequence simultaneously.
- Scaled Dot-Product Attention: The attention mechanism computes attention scores between pairs of words in the sequence using a scaled dot-product. This is done through the formula:

   ![image](https://github.com/user-attachments/assets/07df1d92-33a8-474d-9f37-f0058815fe0a)


## Research Papers

- [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [A Survey of LLMs](https://arxiv.org/pdf/2303.18223.pdf)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf)
