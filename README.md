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
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Results](#results)
7. [Conclusion and Future Work](#conclusion-and-future-work)
8. [Research Papers](#research-papers)
9. [Contact](#contact)

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

1. **Transformer Architecture**:
   - The Transformer architecture, introduced in the “Attention Is All You Need” paper, is the backbone of the GPT model. It replaces recurrent and convolutional layers with a purely attention-based mechanism, enabling the model to handle dependencies between words over long distances without the need for sequential data processing.

![image](https://github.com/user-attachments/assets/0f34fa22-7293-404c-a5dd-6d003a411489)

2. **Self-Attention Mechanism**:
- Self-attention allows the model to weigh the importance of different words in a sequence when encoding a particular word. In GPT, this mechanism helps the model understand the context by considering all the words in the input sequence simultaneously.
- Scaled Dot-Product Attention: The attention mechanism computes attention scores between pairs of words in the sequence using a scaled dot-product. This is done through the formula:

   ![image](https://github.com/user-attachments/assets/07df1d92-33a8-474d-9f37-f0058815fe0a)

  where  Q ,  K , and  V  represent the Query, Key, and Value matrices derived from the input, and  d_k  is the dimension of the keys.

3. **Positional Encoding** :
   - Since the Transformer model doesn’t inherently understand the order of words (because it processes sequences in parallel), positional encodings are added to the input embeddings to give the model information about the position of each word in the sequence.

4. **Decoder-Only Architecture** :
- GPT uses a decoder-only architecture from the Transformer model, which is designed for autoregressive tasks. This means the model predicts the next word in a sequence by conditioning on the previous words.
- The decoder stack is composed of multiple identical layers, each containing two main sub-layers:
- Masked Multi-Head Self-Attention: Prevents the model from looking at future tokens, ensuring that predictions are made only based on past tokens.
- Position-wise Feed-Forward Networks: Applies two linear transformations with a ReLU activation in between.

5. **Pre-training and Fine-tuning** :
- Pre-training: GPT is pre-trained on a large corpus of text in an unsupervised manner, learning to predict the next word in a sequence. This phase allows the model to capture a wide range of language patterns, grammatical structures, and factual knowledge.
- Fine-tuning: After pre-training, the model can be fine-tuned on a smaller dataset specific to a target task, such as text classification, summarization, or dialogue generation. During fine-tuning, the model adapts its pre-learned knowledge to the specific requirements of the task.

6. **Multi-Head Attention** :
- The attention mechanism in GPT is extended to multiple heads, allowing the model to focus on different parts of the input sequence simultaneously. Each attention head operates independently, and their outputs are concatenated and linearly transformed to produce the final output.
- Multi-Head Attention enhances the model’s ability to capture various relationships within the text, improving the richness of the generated content.

**Advantages of GPT Model** :
- Parallel Processing: Unlike RNNs, the Transformer architecture allows for parallel processing of data, significantly speeding up training and inference times.
- Contextual Understanding: The self-attention mechanism enables GPT to capture complex dependencies between words, leading to more contextually accurate text generation.
- Scalability: The GPT model can be scaled up by increasing the number of layers, attention heads, and the size of the hidden states, resulting in models like GPT-2 and GPT-3, which are capable of generating human-like text.

**Limitations** :
- Compute Intensive: The model’s complexity and size require substantial computational resources for training and deployment.
- Data Dependency: The quality and diversity of the training data significantly impact the model’s performance, particularly in generating coherent and factually correct text.

**Use Cases** :
- Text Generation: GPT is widely used for generating human-like text for chatbots, content creation, and storytelling.
- Summarization: It can generate concise summaries of long documents while maintaining the essential information.
- Translation: Although not primarily designed for translation, GPT can perform translation tasks by being fine-tuned on parallel corpora.

**Implementation** :
Implemented in the **gpt-v1.ipynb** file.

## Features

- **Customizable Architecture**: Easily adjust model parameters to explore different configurations.
- **Data Handling**: Efficiently process and handle text data for training and evaluation.
- **Training and Evaluation**: Scripts provided for training both models and evaluating their performance.
- **Visualization**: Tools for visualizing training progress and model outputs.

## Installation

### Prerequisites
- Python 3.8 or higher
- Availability of parallelism, CUDA if you have an NVIDIA GPU or Metal FX (MPS) if you're running Apple Silicon.
- pytorch
- numpy
- matplotlib
- pylzma
- ipykernel
- jupyter

### Installation steps:

1. **Clone the repository**:

   `git clone https://github.com/jayeshCodes/LLM-Forge.git`

   `cd LLM-Forge`

2. **Create Virtual Environment**:

   `python3 -m venv llm-env`

   `source llm-env/bin/activate`  
   On Windows, use `llm-env\Scripts\activate`

3. **Install the Requirements** :

   (Assuming MacOS) `pip3 install numpy pylzma ipykernel jupyter matplotlib torch torchvision torchaudio`


## Usage

1. To interact with the Bigram Model, run `bigram.ipynb`

2. To train the GPT Model:
      `python train.py`

3. To interact with the Chatbot:
      `python chatbot.py`

## Dataset

1. For the Bigram Model, the novel `The Wizard of Oz` was used from `https://www.gutenberg.org/ebooks/55` in plain `utf-8` txt format.
2. For the GPT model,
   
      `https://huggingface.co/datasets/Skylion007/openwebtext/tree/main` dataset was used.
   
      Run the `dataextract.py` script to **extract the data as txt** from the .xz files,
   
      To **clean the txt files**, run the `dataclean.py` script to **remove all the non-English** characters in the txt files.

## Results

1. Due to limitations in time, data as well as resources, the models have not been trained as extensively as the outgoing versions of ChatGPT.
2. This results in non-comprehensible results to humans. A few examples from the Bigram as well as GPT models are listed below.
3. Examples:
   - **Bigram Model** :
     
     ![image](https://github.com/user-attachments/assets/326c10e0-4ba4-4677-821e-fa0d09eeb324)

     Model loss : 2.386
     
     Number of epochs : 10000

   - **GPT Model** :
     
     ![image](https://github.com/user-attachments/assets/b95b00b5-5d76-44b5-8567-42e1cae044be)

     ![image](https://github.com/user-attachments/assets/28a1a092-1287-4e14-964c-d6e7ef57aace)

     Model loss : 1.985

     After being trained twice on the same dataset.

## Conclusion and Future Work

### Conclusion

This project was undertaken as an effort to delve deeper into the ever evolving world of Language models and to discover the intensiveness of creating a Large Language Model from scratch.
As such, the GPT model doesn't perform well and needs some more work which will be discussed below.

### Future Work

To help the GPT model perform even better, I believe that the following steps should be taken as the future work of this project:

1. **Training** the model extensively:

   The first and most obvious improvement that can be made to this model, is training it extensively. Currently, it has been trained only for approximately 6000 epochs, which is very less when we realise that the commercial GPT models have been trained for several months. Training this model more extensively and for longer periods of time will help the model form better relations between the tokens and help it make better predictions which will in turn result in better language generation.

2. **Auto-tuning** the hyperparameters:

   Playing around with the hyperparameters and implementing an auto-tuning script which tunes the hyperparameters automatically will greatly boost the learning capabilities of the GPT model. The current hyperparameters being used in this version of the model were chosen after trial and error by myself which is not be the most optimal solution.

3. Adding a script to **Evaluate** the model on more parameters rather than just loss:

   The current version of the model only takes into account loss as a parameter for evaluation. Future iterations may greatly benefit from the addition of an `evaluation.py` scripts which evaluates each model iteration's performance metrics.

4. **Fine-tuning** : This is beyond the scope of this project, but fine-tuning may greatly improve the performance of the current GPT model (coming up soon in future projects).


## Research Papers

The following research papers greatly helped me in understanding the inner workings and mechanisms of how a LLM works, and in turn helped me in creating this project. I am extremely grateful to all the researchers whose work has enabled me to build this project.

- [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [A Survey of LLMs](https://arxiv.org/pdf/2303.18223.pdf)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf)

## Contact

For questions, suggestions, or collaborations, feel free to contact me:

   Email : jaygajbhar@gmail.com

   LinkedIn : [Jayesh Gajbhar](https://www.linkedin.com/in/jayeshgajbhar/)
