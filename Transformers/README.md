# Transformer From Scratch in PyTorch

This repository documents my journey of learning and implementing the Transformer architecture from scratch using PyTorch.

The goal is **not just to use Transformers**, but to understand every component mathematically, intuitively, and through implementation.

Before moving into modern architectures such as Vision Transformers (ViT), MAE, DINO, JEPA, and Large Language Models, I want to build a solid understanding of the original Transformer architecture introduced in the paper:

**Attention Is All You Need (2017)**

---

# Learning Goals

By the end of this project, I aim to:

* Understand how attention works
* Implement Transformer components from scratch
* Learn tensor shapes throughout the architecture
* Build a complete Transformer Encoder
* Build a complete Transformer Decoder
* Understand BERT, GPT, and ViT architectures
* Develop intuition for modern self-supervised learning methods

---

# Learning Roadmap

## Phase 0 — Prerequisites

Before learning Transformers, I reviewed:

* Linear Layers
* Matrix Multiplication
* Softmax
* Embeddings
* Residual Connections
* Normalization Layers

---

## Phase 1 — Attention Fundamentals

### 1. Query, Key, Value (QKV)

Understanding:

* What Queries represent
* What Keys represent
* What Values represent
* Why attention requires all three

### 2. Dot Product Similarity

Learning how tokens measure similarity with one another.

### 3. Scaled Dot Product Attention

Core Transformer operation:

Attention(Q, K, V)

Understanding:

* Similarity scores
* Scaling factor
* Softmax normalization
* Weighted aggregation

### 4. Single-Head Self-Attention

Implementation from scratch using:

* nn.Linear
* torch.matmul
* torch.softmax

---

## Phase 2 — Multi-Head Attention

### 5. Head Splitting

Understanding:

* Why multiple heads are used
* How embedding dimensions are split

### 6. Multi-Head Attention

Pipeline:

Input
→ QKV Projection
→ Split Heads
→ Attention
→ Concatenate
→ Output Projection

Implementation from scratch.

---

## Phase 3 — Transformer Encoder Block

### 7. Residual Connections

Understanding:

Input + Attention Output

### 8. Layer Normalization

Learning:

* Why LayerNorm is used
* Why BatchNorm is avoided

### 9. Feed Forward Network (FFN)

Architecture:

Linear
→ Activation
→ Linear

### 10. Complete Encoder Block

Pipeline:

Input
→ Multi-Head Attention
→ Add & Norm
→ Feed Forward
→ Add & Norm
→ Output

Implementation from scratch.

---

## Phase 4 — Positional Information

### 11. Why Position Matters

Understanding why attention alone cannot understand sequence order.

### 12. Sinusoidal Positional Encoding

Learning:

* Sine functions
* Cosine functions
* Frequency-based position representation

### 13. Learned Positional Embeddings

Modern approach used by many Transformer models.

---

## Phase 5 — Full Transformer Encoder

Building:

Token Embeddings
+
Position Embeddings
↓
N Encoder Blocks
↓
Output Representations

At this stage, the architecture becomes very similar to:

* BERT
* ViT Encoder
* JEPA Encoders

---

## Phase 6 — Transformer Decoder

### 14. Causal Masking

Preventing future token information leakage.

### 15. Masked Self-Attention

Understanding autoregressive prediction.

### 16. Cross-Attention

Learning how encoder outputs interact with decoder inputs.

---

## Phase 7 — Complete Transformer

Implementing the original architecture:

Encoder
↓
Cross Attention
↓
Decoder

Based on:

Attention Is All You Need (2017)

---

## Phase 8 — Modern Transformer Variants

After understanding the original architecture:

### BERT

Encoder-only Transformer

### GPT

Decoder-only Transformer

### Vision Transformer (ViT)

Image
↓
Patch Embedding
↓
Transformer Encoder

### MAE

Masked Autoencoder

### DINO

Self-Distillation with No Labels

### JEPA

Joint Embedding Predictive Architecture

---

# Repository Structure

```text
Transformer-From-Scratch/

├── 01_QKV.ipynb
├── 02_Dot_Product_Similarity.ipynb
├── 03_Scaled_Dot_Product_Attention.ipynb
├── 04_Single_Head_Attention.ipynb
├── 05_Multi_Head_Attention.ipynb
├── 06_Residual_Connections.ipynb
├── 07_LayerNorm.ipynb
├── 08_FeedForward_Network.ipynb
├── 09_Encoder_Block.ipynb
├── 10_Positional_Encoding.ipynb
├── 11_Transformer_Encoder.ipynb
├── 12_Masking.ipynb
├── 13_Decoder.ipynb
├── 14_Cross_Attention.ipynb
├── 15_Full_Transformer.ipynb
├── README.md
```

---

# Tech Stack

* Python
* PyTorch
* Jupyter Notebook
* NumPy
* Matplotlib

---

# Learning Philosophy

This repository focuses on:

* Understanding over memorization
* Building before using
* Visualizing tensor operations
* Learning through implementation

Every concept is implemented from scratch before using PyTorch's high-level abstractions.

---

# Future Work

After completing this repository, I plan to continue with:

1. Vision Transformer (ViT)
2. MAE
3. DINO
4. JEPA
5. Large Language Models
6. Multimodal Foundation Models

---

# Acknowledgements

* Attention Is All You Need (2017)
* The Illustrated Transformer
* PyTorch Documentation
* Open-source AI Research Community

---

This repository is part of my ongoing journey to deeply understand modern deep learning architectures by implementing them from first principles.
