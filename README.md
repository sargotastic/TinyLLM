# TinyGPT — A Small Language Model Trained from Scratch

**Author:** Sarthak Goyal
**Framework:** PyTorch
**Tokenizer:** SentencePiece
**Keywords:** Transformer, NLP, Language Model, GPT, Deep Learning

---

## Overview

TinyGPT is a minimal Transformer-based language model that is implemented and trained from scratch using PyTorch and SentencePiece.
The goal of this project is to demonstrate a practical understanding of how large language models (LLMs), such as GPT, work internally, from tokenization and embeddings to attention and text generation.

The model is a small, single-machine implementation of the core GPT architecture, capable of learning grammar and producing coherent text when trained on small datasets such as BookCorpus or TinyStories.

---

## Features

* Custom tokenizer trained with SentencePiece
* Transformer decoder with multi-head self-attention
* Causal masking for autoregressive training
* Configurable architecture (layers, heads, embedding size)
* Simple training pipeline using PyTorch
* Text generation with temperature and token sampling
* Compatible with CPU and Apple Silicon GPU acceleration

---

## Architecture

The model follows a standard GPT-like structure:

```
Text → Tokenization → Embeddings → Positional Encoding
     → Multi-Head Self-Attention → Feed-Forward Layers
     → Softmax → Next Token Prediction
```

**Key Components:**

* Token and positional embeddings
* Multi-head scaled dot-product self-attention
* Feed-forward neural network with ReLU activation
* Residual connections and layer normalization
* Cross-entropy loss for next-token prediction

Example configuration:

```python
config = GPTConfig(
    vocab_size=5000,
    n_layers=6,
    n_heads=6,
    d_model=384,
    d_ff=1536,
    context_length=512,
    dropout=0.1
)
```

---

## Project Structure

```
tiny-llm/
│
├── data/                # Training data (.txt)
├── tokenizer/           # SentencePiece tokenizer files
├── model/
│   └── tiny_gpt.py      # Transformer model implementation
├── train_dataset.py     # Dataset loader
├── train_tokenizer.py   # Tokenizer training script
├── train.py             # Training loop
├── generate.py          # Text generation script
└── requirements.txt     # Python dependencies
```

---

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install torch tqdm numpy sentencepiece matplotlib
```

---

## Data Preparation

Place a plain-text dataset at:

```
data/raw.txt
```

Recommended datasets:

* [BookCorpus](https://huggingface.co/datasets/bookcorpus)
* [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
* Public-domain literature (e.g., Sherlock Holmes, Pride and Prejudice)

---

## Training

Train the tokenizer:

```bash
python train_tokenizer.py
```

Train the model:

```bash
python train.py
```

After training, a model checkpoint will be saved as:

```
checkpoint.pt
```

A loss curve will also be generated and saved as `training_loss.png`.

---

## Model Checkpoint

The trained weights (`checkpoint.pt`) are not included in this repository due to file size limitations.
To reproduce the checkpoint, run the training script above using your own dataset.

If needed, a trained checkpoint can be shared externally (for example, through Google Drive or Hugging Face Hub).

---

## Text Generation

After training, generate text using:

```bash
python generate.py
```

Example:

```
Enter a prompt: The sun rose over the city
```

Output:

```
The sun rose over the city and the streets began to shimmer with quiet light.
```

The output length and creativity can be controlled with parameters in `generate.py`:

* `MAX_NEW_TOKENS`: number of generated tokens
* `TEMPERATURE`: randomness (lower = focused, higher = creative)

---

## Results

| Dataset                       | Description               | Output Style       |
| ----------------------------- | ------------------------- | ------------------ |
| Meditations (Marcus Aurelius) | Philosophical reflections | Stoic, reflective  |
| BookCorpus                    | General modern English    | Fluent, coherent   |
| TinyStories                   | Synthetic English stories | Structured, simple |

Training time: approximately 2–4 hours on Apple M2 hardware.

---

## How It Works

1. **Tokenization** – The SentencePiece tokenizer converts text into subword tokens.
2. **Embeddings** – Each token is mapped into a numerical vector space.
3. **Positional Encoding** – Adds positional context to each token embedding.
4. **Self-Attention** – Each token attends to previous tokens to learn relationships.
5. **Feed-Forward Network** – Processes attention outputs to refine representations.
6. **Softmax Layer** – Predicts the probability of the next token.
7. **Training Objective** – Minimizes cross-entropy loss for next-token prediction.
8. **Generation Loop** – Samples tokens autoregressively to produce text sequences.

---

## Example Run

```bash
# Activate virtual environment
source /Users/sarthakgoyal/Documents/tiny-llm/.venv/bin/activate

# Train the model
(.venv) sarthakgoyal@Sarthak tiny-llm % python train.py
Epoch 1/3: 100%|██████████████████████████████████████████████████████████████████| 12631/12631 [3:03:26<00:00, 1.15it/s, loss=0.0935]
Epoch 2/3: 100%|██████████████████████████████████████████████████████████████████| 12631/12631 [3:03:40<00:00, 1.15it/s, loss=0.0618]
Epoch 3/3: 100%|██████████████████████████████████████████████████████████████████| 12631/12631 [3:09:26<00:00, 1.11it/s, loss=0.0698]
Model training complete! Saved as checkpoint.pt

# Generate text
(.venv) sarthakgoyal@Sarthak tiny-llm % python generate.py
Model loaded! Ready to generate.

Enter a prompt: The quick brown fox

Generated text:
-----------------------------------
The quick brown fox was scared and started to cry. The bear ran away. Even though the bear had taken the snow he had not had a great day after all.  ⁇ endoftext ⁇  Molly and her mom were sitting outside in the sun. It was Wednesday and they were enjoying their healthy snack.  ⁇ Look what I made, ⁇  said mom. Molly looked over
-----------------------------------

You: I am a boy
tinyLLM: I am a boy named Timmy. Now, he could use some help. Now, he did. He looked back and saw a big green frog with big eyes. The frog saw the squirrel was lost, so it jumped to the squirrel ⁇ s aid. The frog saw the squirrel needed help to find its way home, so it supplied a few friendly hops and jumps to show

You: story time
tinyLLM: story time to go home. ⁇  The two friends started to playing the weapon. They ran and jumped and rolled on the grass. Ben watched them from the tree. He felt lonely and sad. Then, something bad happened. The dog saw a squirrel. The dog wanted to catch the squirrel. The dog ran after the squirrel. The dog forgot about Lily. The dog pull

You: green
tinyLLM: green park with her family. She, curious and excited to explore the unknown. As she looked down at her staff and hopped back and forth, planting each flower with a big splash. In the end, the little girl ⁇ s mama was very proud of her daughter, and she thanked her for all the hard work. The little girl smiled, knowing that she had done

You: red ball
tinyLLM: red balloon. That made it easy to pass.  ⁇ oe quickly grabbed the handle and passed the balloon to his friend. His friend was so happy and they played with the balloon until the sun went down!  ⁇ endoftext ⁇  Once upon a time, there was a light that shone in the darkness. One night, something very sad happened. The light became miserable and wasn ⁇ t so bright

You: safe
tinyLLM: safe. He was so excited to have done something so special and was looking forward to doing it again!  ⁇ endoftext ⁇  It was a hot summer day. Lucy and her mom were out in the yard. Lucy ⁇ s mom was gardening and Lucy was playing with a ball. Suddenly, Lucy ⁇ s ball rolled towards the house and stop
```

## Future Work

* Visualize attention weights
* Experiment with weight quantization (INT8)
* Add interactive chat interface
* Fine-tune on domain-specific datasets
* Export model to ONNX or CoreML for deployment

---

## License

This project is released under the MIT License.

---

## Citation

If referencing this project, please cite as:

```
Goyal, Sarthak. (2025). TinyGPT: A Small Transformer Language Model Trained from Scratch. GitHub repository.
```
