# Build an LLM

<img width="847" height="660" alt="Screenshot 2025-12-24 at 20 57 51" src="https://github.com/user-attachments/assets/2fbb32ac-45bd-4b27-9321-79dc65e14242" />

This repository contains a complete, educational implementation of a transformer-based autoregressive, decoder-only language model.

It offers two learning objectives:

- **Pre-train an LLM from scratch** using a simple, intuitive interface.
- **Explore the codebase** to understand the modularized building blocks of transformer models, with multiple implementation variants for each component. 

I built this project as I wanted to properly understand LLMs. To do this, I had to write code myself rather than just watch videos and read books and lecture notes ((links below)[(#resources)]). I'm incredibly grateful to all those from whom I learned and borrowed ideas. I hope others find it helpful too!

_(This repository is a work in progress. Comments, corrections, and pull requests are very welcome. Currently, it serves a decoder-only architecture (like GPT, LLaMA, OLMo) and does not include encoder-only models (like BERT), encoder-decoder models (like T5), or Mixture of Experts architectures (like DeepSeek-V2). It also does not yet go past the pre-training stage.)_

## What You'll Learn

By exploring this codebase, you'll gain a deep understanding of:

- **Transformer Architecture**: How decoder-only language models work from the ground up
- **Core Components**: Attention mechanisms, normalization layers, positional encodings, and feedforward networks
- **Architecture Variants**: Differences between GPT, LLaMA, and OLMo implementations
- **Training Process**: How to pre-train a language model using next-token prediction
- **Text Generation**: Autoregressive generation and sampling strategies (temperature, top-k, top-p)
- **Implementation Details**: Multiple implementation approaches (with/without einops, manual vs PyTorch built-ins) to understand what's happening under the hood

The codebase includes both educational implementations (showing the math and operations explicitly) and optimized versions, so you can see how concepts translate to actual code.

## Contents

1. [Usage Guide](#usage-guide)
2. [Key Concepts](#key-concepts)
3. [Pre-Training Pipeline](#pre-training-pipeline)
4. [Inference and Sampling](#inference-and-sampling)
5. [Core Components Deep Dive](#core-components-deep-dive)
6. [Resources](#resources)

---

## Usage Guide

### Getting Started

This project is a **Streamlit web application** that provides an interactive interface for training and inference.

**Start the application:**
```bash
uv run --with streamlit streamlit run main.py
```

The app will open in your browser with the following pages:
- **Main**: Overview and this README
- **Pre-Training Page**: Configure and pre-train models with a visual interface
- **Inference Page**: Generate text from trained models

### Pre-Training

**Using the Streamlit UI:**

1. Upload training data or use the default `training.txt` file
2. Select an architecture preset, or choose custom parameters:
   - **🚀 GPT-2**: Learned positional embeddings, LayerNorm, GELU activation
   - **🦙 LLaMA**: RoPE positional encoding, RMSNorm, SwiGLU activation
   - **🔬 OLMo**: ALiBi positional encoding, LayerNorm, SwiGLU activation
3. Configure model dimensions (or use size presets: small, medium, full)
4. Set training hyperparameters (batch size, learning rate, epochs, etc.)
5. Click "Start Training" to begin

**What happens**:
1. Loads training text file
2. Creates tokenizer and dataset
3. Initializes model based on selected architecture and configuration
4. Trains for specified epochs with real-time loss visualization
5. Saves checkpoints to `checkpoints/YYYYMMDDHHMMSS/` (timestamped folders)

**Tokenizer Types**:
- **Character-level**: Each character is a token. Simple but large vocabulary.
- **BPE-Simple**: Learns subword units. Good balance of vocabulary size and efficiency.
- **BPE-TikToken**: Learns subword units. Good balance of vocabulary size and efficiency.
- **SentencePiece**: Similar to BPE but handles whitespace differently. Often used in multilingual models.

**Command-line Training:**

You can also train models using the command-line script:
```bash
# Train with default settings (GPT, small, einops, character tokenizer)
uv run cli/train.py

# Train LLaMA model
uv run cli/train.py --architecture LLAMA

# Train OLMo model
uv run cli/train.py --architecture OLMO

# Train full-size model
uv run cli/train.py --model_size full

# Train with BPE tokenizer
uv run cli/train.py --tokenizer_type bpe

# Train without einops
uv run cli/train.py --no_einops
```

### Inference

**Using the Streamlit UI:**

1. Select a checkpoint from the dropdown (auto-scans `checkpoints/` directory)
2. Enter a prompt
3. Configure sampling parameters (temperature, top-k, top-p)
4. Click "Generate" to create text

**Command-line Inference:**

```bash
# Generate text from trained model
uv run cli/infer.py --checkpoint checkpoints/20240101120000/final_model.pt --prompt "First Citizen:"
```

**Options**:
- `--checkpoint`: Path to model checkpoint (e.g., `checkpoints/20240101120000/final_model.pt`)
- `--prompt`: Starting text prompt
- `--max_new_tokens`: Number of tokens to generate (default: 200)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top_k`: Top-k sampling (optional)
- `--top_p`: Top-p sampling (optional, default: 0.9)
- `--tokenizer_type`: Tokenizer type (optional, auto-detected from checkpoint; only needed for old checkpoints)
- `--text_file`: Text file for character tokenizer initialization (default: `training.txt`, only needed for character tokenizer)

---

## Key Concepts

Before diving into the implementation details, here are the key concepts that underpin this codebase:

### 1. Autoregressive Language Modeling

**What it is**: Predicting the next token given previous tokens in a sequence.

**Example**:
```
Input:  "The cat sat on the"
Target: "cat sat on the mat"
```

At each position, the model predicts what comes next. This is how language models learn to generate coherent text.

**Why it works**: 
- Language has structure and patterns
- Given context, the next token is somewhat predictable
- The model learns these patterns from data

### 2. Causal Masking

**What it is**: Preventing the model from seeing future tokens during training.

**Why it's needed**:
- During inference, we generate one token at a time
- The model should only use past context
- Training must match inference conditions

**How it works**:
- Attention scores for future positions are set to `-inf`
- After softmax, these become 0 probability
- The model can't attend to future tokens

### 3. Residual Connections

**What it is**: Adding input to output: `output = input + transformation(input)`

**Why it helps**:
- Allows gradients to flow directly through
- Enables training of very deep networks
- The model can learn the identity function if the transformation isn't needed

### 4. Layer Normalization

**What it is**: Normalizing activations across the feature dimension.

**Why it helps**:
- Stabilizes training
- Allows higher learning rates
- Reduces internal covariate shift

**Variants**:
- **LayerNorm** (GPT/OLMo): Normalizes by subtracting mean, then scaling
- **RMSNorm** (LLaMA): Only scales (no mean subtraction, no bias)

### 5. Positional Encoding

**The Problem**: Transformers have no inherent notion of sequence order.

**Solutions**:
- **Learned Embeddings** (GPT): Fixed embeddings for each position
- **RoPE** (LLaMA): Rotates query/key vectors by position-dependent angles
- **ALiBi** (OLMo): Adds distance-based bias to attention scores

### 6. Pre-training vs Fine-tuning

**Pre-training** (what this codebase does):
- Train on large, diverse text corpus
- Learn general language patterns
- Unsupervised (no labels needed)

**Fine-tuning** (not included):
- Take pre-trained model
- Train further on specific task/domain
- Supervised (needs labeled data)

---

## Pre-Training Pipeline

### 1. Data Loading (`pretraining/data/dataset.py`)

**Purpose**: Load text, tokenize, and create training sequences.

#### Process

1. **Load Text**: Read raw text file
2. **Tokenize**: Convert text to token IDs
3. **Create Sequences**: Sliding window approach
   - Input: `[token_0, token_1, ..., token_n-1]`
   - Target: `[token_1, token_2, ..., token_n]` (shifted by 1)
4. **Split**: Train/validation split (default 90/10)

#### Code Flow

```python
# Load text
text = "Hello world..."

# Tokenize
data = tokenizer.encode_tensor(text)  # [total_tokens]

# Create sequences (sliding window)
for i in range(len(data) - block_size):
    X.append(data[i:i+block_size])      # Input
    Y.append(data[i+1:i+block_size+1])  # Target (shifted)

# X: [num_sequences, block_size]
# Y: [num_sequences, block_size]
```

**Why Shift by 1?**
- We predict the next token
- At position `i`, we predict token at position `i+1`
- This is autoregressive language modeling

### 2. Training Loop (`pretraining/training/trainer.py`)

**Purpose**: Train the model using gradient descent.

#### Key Components

1. **Optimizer**: AdamW (adaptive learning rate with weight decay)
2. **Loss Function**: Cross-entropy (next-token prediction)
3. **Evaluation**: Periodic evaluation on train/val sets
4. **Checkpointing**: Save model periodically

#### Training Step

```python
# 1. Get random batch
idx = torch.randint(0, len(X_train), (batch_size,))
x_batch = X_train[idx]  # [batch_size, seq_len]
y_batch = Y_train[idx]  # [batch_size, seq_len]

# 2. Forward pass
logits = model(x_batch)  # [batch_size, seq_len, vocab_size]

# 3. Compute loss
# Reshape for cross-entropy
logits_flat = logits.view(-1, vocab_size)  # [batch*seq, vocab]
targets_flat = y_batch.view(-1)  # [batch*seq]
loss = F.cross_entropy(logits_flat, targets_flat)

# 4. Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**Loss Explanation**:
- Cross-entropy measures how well predicted probabilities match true token
- Lower loss = better predictions
- Typical range: starts high (2-4), decreases during training

**Evaluation**:
- Run model in `eval()` mode (no dropout, no gradient computation)
- Average loss over multiple random batches
- Compare train vs val loss to detect overfitting

### 3. Configuration (`config.py`, `pretraining/training/training_args.py`)

#### `ModelConfig` - Model Architecture
- `d_model`: Hidden dimension (e.g., 256, 768)
- `n_layers`: Number of transformer blocks (e.g., 4, 12)
- `n_heads`: Number of attention heads (e.g., 4, 12)
- `d_head`: Dimension per head (typically `d_model / n_heads`)
- `d_mlp`: MLP hidden dimension (typically `4 * d_model`)
- `n_ctx`: Context length (max sequence length)
- `d_vocab`: Vocabulary size
- `architecture`: Architecture type (GPT, LLaMA, or OLMo)

#### `TransformerTrainingArgs` - Training Hyperparameters
- `batch_size`: Number of sequences per batch
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `weight_decay`: L2 regularization strength
- `eval_iters`: Number of batches for evaluation

---

## Inference and Sampling

### Text Generation (`inference/sampler.py`)

**Purpose**: Generate text from a trained model.

#### Autoregressive Generation

```python
# Start with prompt
tokens = tokenizer.encode(prompt)  # [seq_len]

# Generate tokens one by one
for _ in range(max_new_tokens):
    # Get model predictions
    logits = model(tokens)  # [1, seq_len, vocab_size]
    
    # Get logits for last position
    next_token_logits = logits[0, -1, :]  # [vocab_size]
    
    # Sample next token
    next_token = sample(next_token_logits)
    
    # Append to sequence
    tokens.append(next_token)
```

#### Sampling Strategies

1. **Temperature Sampling**:
   ```python
   logits = logits / temperature
   probs = softmax(logits)
   ```
   - `temperature < 1`: More focused (deterministic)
   - `temperature = 1`: Balanced
   - `temperature > 1`: More creative (random)

2. **Top-k Sampling**:
   ```python
   # Only consider top k most likely tokens
   top_k_logits = topk(logits, k)
   ```
   - Prevents sampling very unlikely tokens
   - `k=40` is common

3. **Top-p (Nucleus) Sampling**:
   ```python
   # Consider tokens until cumulative probability > p
   sorted_probs = sort(softmax(logits))
   cumulative = cumsum(sorted_probs)
   mask = cumulative <= p
   ```
   - Adaptive: considers more tokens when distribution is flat
   - `p=0.9` is common

4. **Combined**: Top-k + Top-p + Temperature
   - Most common in practice
   - Provides good balance of quality and diversity

---

## Core Components Deep Dive

### 1. Normalization Layers

#### Layer Normalization (`pretraining/normalization/layernorm.py`) - GPT/OLMo Style

**Purpose**: Normalize activations across the feature dimension to stabilize training.

**Three Implementations:**

#### `LayerNormWithEinops`
Uses `einops.reduce` for mean and variance computation:
```python
# Compute mean: [batch, posn, d_model] -> [batch, posn, 1]
residual_mean = einops.reduce(
    residual, 'batch posn d_model -> batch posn 1', 'mean'
)
```

**Why this works**: Einops makes it explicit that we're reducing over `d_model` while keeping `batch` and `posn`.

#### `LayerNormWithoutEinops`
Uses PyTorch's built-in operations:
```python
# Compute mean: [batch, posn, d_model] -> [batch, posn, 1]
residual_mean = residual.mean(dim=-1, keepdim=True)
```

**Why this works**: `dim=-1` means "last dimension" (d_model), `keepdim=True` preserves the dimension.

#### `LayerNormWithTorch`
Uses PyTorch's `nn.LayerNorm`:
```python
self.ln = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
```

**Why this works**: PyTorch's implementation is optimized and handles edge cases.

**Mathematical Formula:**
```
LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
```
Where:
- `μ` = mean over d_model dimension
- `σ` = standard deviation over d_model dimension
- `γ` (w) = learnable scale parameter
- `β` (b) = learnable shift parameter
- `ε` = small constant for numerical stability

**Shape Flow:**
```
Input:  [batch, posn, d_model]
Mean:   [batch, posn, 1]        # Mean over d_model
Std:    [batch, posn, 1]        # Std over d_model
Normalized: [batch, posn, d_model]
Output: [batch, posn, d_model]  # After scale and shift
```

---

#### RMS Normalization (`pretraining/normalization/rmsnorm.py`) - LLaMA Style

**Purpose**: Simpler normalization used in LLaMA (no mean subtraction, no bias).

**Key Difference from LayerNorm:**
- LayerNorm: `(x - mean) / std * γ + β` (centers then scales)
- RMSNorm: `x / rms * γ` (only scales, no centering, no bias)

**Implementation:**
```python
# Compute RMS (Root Mean Square)
rms = sqrt(mean(x²) + eps)  # [batch, posn, 1]

# Normalize
norm = x / rms  # [batch, posn, d_model]

# Apply scale (no bias)
output = norm * w  # [batch, posn, d_model]
```

**Why RMSNorm?**
- Simpler (fewer operations)
- No bias term needed
- Works well in practice
- Used in LLaMA, PaLM, and other modern models

**Shape Flow:**
```
Input:  [batch, posn, d_model]
RMS:    [batch, posn, 1]        # RMS over d_model
Normalized: [batch, posn, d_model]
Output: [batch, posn, d_model]  # After scale only
```

---

### 2. Token Embeddings (`pretraining/embeddings/embed.py`)

**Purpose**: Convert token IDs (integers) into dense vector representations.

#### `EmbedWithoutTorch`
Manual implementation using `nn.Parameter`:
```python
# W_E: [d_vocab, d_model] - embedding matrix
self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))

# Forward: Index into embedding matrix
# tokens: [batch, position] -> [batch, position, d_model]
return self.W_E[tokens]
```

**How it works**: 
- Each token ID (0 to d_vocab-1) maps to a row in `W_E`
- `W_E[tokens]` uses advanced indexing to look up embeddings
- Example: `tokens = [[5, 10, 3]]` → `[W_E[5], W_E[10], W_E[3]]`

#### `EmbedWithTorch`
Uses PyTorch's `nn.Embedding`:
```python
self.embedding = nn.Embedding(cfg.d_vocab, cfg.d_model)
return self.embedding(tokens)
```

**Why both versions**: 
- Manual version shows what's happening under the hood
- PyTorch version is optimized and handles edge cases

**Shape Flow:**
```
Token IDs: [batch, position]  # e.g., [[5, 10, 3]]
Embedding Matrix: [d_vocab, d_model]  # e.g., [50257, 256]
Output: [batch, position, d_model]  # e.g., [[emb[5], emb[10], emb[3]]]
```

---

### 3. Positional Encoding

#### Learned Positional Embeddings (`pretraining/positional_embeddings/positional_embedding.py`) - GPT Style

**Purpose**: Add information about token positions in the sequence.

#### `PosEmbedWithEinops`
```python
# W_pos: [n_ctx, d_model] - one embedding per position
# Get embeddings for current sequence length
W_pos[:seq_len]  # [seq_len, d_model]

# Repeat for each item in batch
einops.repeat(
    W_pos[:seq_len], 
    "seq d_model -> batch seq d_model", 
    batch=batch
)
```

**How it works**:
- `W_pos[i]` is the embedding for position `i`
- `einops.repeat` broadcasts `[seq_len, d_model]` to `[batch, seq_len, d_model]`

#### `PosEmbedWithoutEinops`
```python
# Manual broadcasting
position_embeddings_we_need = self.W_pos[:sequence_length]  # [seq_len, d_model]
position_embeddings_with_batch_dim = position_embeddings_we_need.unsqueeze(0)  # [1, seq_len, d_model]
position_embeddings_repeated = position_embeddings_with_batch_dim.expand(batch_size, -1, -1)  # [batch, seq_len, d_model]
```

**How it works**:
- `unsqueeze(0)` adds a batch dimension at position 0
- `expand()` repeats along batch dimension (memory-efficient, no copying)

**Shape Flow:**
```
W_pos: [n_ctx, d_model]  # e.g., [1024, 256]
Slice: [seq_len, d_model]  # e.g., [128, 256]
Repeat: [batch, seq_len, d_model]  # e.g., [32, 128, 256]
```

**Why positional embeddings?**
- Transformers have no inherent notion of sequence order
- Positional embeddings encode "this token is at position 5"
- Added to token embeddings: `final_emb = token_emb + pos_emb`

#### Rotary Position Embedding (`pretraining/positional_embeddings/rope.py`) - LLaMA Style

**Purpose**: Encode positions through rotations of query and key vectors (not learned, computed on-the-fly).

**Key Concepts:**
- **Not added to embeddings**: Applied directly to Q and K in attention
- **Rotation-based**: Rotates each dimension pair by position-dependent angles
- **Relative positions**: Encodes relative distances between tokens

**How it works:**
1. Split Q/K into pairs: `[d_head] → [d_head/2 pairs]`
2. For each pair `(x_i, x_i+1)`, compute rotation angle: `θ_i * position`
3. Apply rotation matrix:
   ```
   [cos(θ)  -sin(θ)]  [x_i  ]
   [sin(θ)   cos(θ)]  [x_i+1]
   ```
4. Different frequencies for different dimensions: `θ_i = 10000^(-2i/d_head)`

**Implementation:**
```python
# Pre-compute frequencies
freqs = 1.0 / (theta ** (arange(0, d_head, 2) / d_head))

# For each position, compute rotation angles
angles = positions * freqs  # [seq, d_head/2]

# Rotate Q and K
q_rotated, k_rotated = apply_rotation(q, k, angles)
```

**Why RoPE?**
- Encodes relative positions naturally
- Extrapolates to longer sequences better than learned embeddings
- Applied in attention (not embeddings), so more flexible
- Used in LLaMA, PaLM, and other modern models

**Shape Flow:**
```
Q/K: [batch, seq, n_heads, d_head]
  ↓
Reshape to pairs: [batch, seq, n_heads, d_head/2, 2]
  ↓
Rotate each pair: [batch, seq, n_heads, d_head/2, 2]
  ↓
Reshape back: [batch, seq, n_heads, d_head]
```

#### ALiBi - Attention with Linear Biases (`pretraining/positional_embeddings/alibi.py`) - OLMo Style

**Purpose**: Encode positions through linear biases added to attention scores (not learned, computed on-the-fly).

**Key Concepts:**
- **Not added to embeddings**: Applied directly to attention scores
- **Distance-based**: Bias depends on distance between positions
- **Per-head slopes**: Each attention head gets a different slope
- **No learned parameters**: All computed from fixed formulas

**How it works:**
1. Compute distance matrix: `distance[i, j] = |i - j|`
2. For each head `h`, compute slope: `slope[h] = 2^(-8/n_heads * h)`
3. Apply bias: `bias[h, i, j] = -slope[h] * distance[i, j]` for future positions
4. Add bias to attention scores before softmax

**Implementation:**
```python
# Pre-compute slopes for each head
slopes = 2^(-8/n_heads * [1, 2, ..., n_heads])  # [n_heads]

# Compute distance matrix
distance = |pos_i - pos_j|  # [seq_len, seq_len]

# Apply slopes
bias = -slopes.unsqueeze(-1).unsqueeze(-1) * distance.unsqueeze(0)  # [n_heads, seq_len, seq_len]

# Add to attention scores
attn_scores = attn_scores + bias
```

**Why ALiBi?**
- Simpler than RoPE (no rotations, just addition)
- Extrapolates extremely well to longer sequences
- Each head learns different distance preferences
- Used in OLMo and other modern models
- No learned parameters, so more efficient

**Shape Flow:**
```
Attention scores: [batch, n_heads, seq_len, seq_len]
  ↓
ALiBi bias: [n_heads, seq_len, seq_len]
  ↓
Add bias: [batch, n_heads, seq_len, seq_len]
  ↓
Softmax: [batch, n_heads, seq_len, seq_len]
```

**Key Insight:**
- Closer positions get less negative bias (can attend more)
- Farther positions get more negative bias (attend less)
- This naturally implements causal attention without explicit masking (though we still mask for numerical stability)

---

### 4. Multi-Head Self-Attention (`pretraining/attention/attention.py`)

**Purpose**: Allow tokens to attend to other tokens in the sequence, learning relationships.

#### Key Concepts

**Query (Q), Key (K), Value (V)**:
- **Query**: "What am I looking for?"
- **Key**: "What do I represent?"
- **Value**: "What information do I contain?"

**Attention Mechanism**:
1. Compute attention scores: `Q @ K^T` (how much each token attends to others)
2. Apply causal mask (prevent looking at future tokens)
3. Softmax to get attention probabilities
4. Weighted sum of values: `Attention(Q, K, V) = softmax(QK^T / √d_head) @ V`

#### `AttentionWithEinops`

**Step 1: Compute Q, K, V**
```python
# residual: [batch, posn, d_model]
# W_Q: [n_heads, d_head, d_model]
# q: [batch, posn, n_heads, d_head]
q = einops.einsum(
    residual, self.W_Q,
    "batch posn d_model, n_heads d_head d_model -> batch posn n_heads d_head"
)
```

**What's happening**: For each head, we project the residual into a `d_head`-dimensional space.

**Step 2: Compute Attention Scores**
```python
# q: [batch, posn_q, n_heads, d_head]
# k: [batch, posn_k, n_heads, d_head]
# attn_scores: [batch, n_heads, posn_q, posn_k]
attn_scores = einops.einsum(
    q, k,
    "batch posn_q n_heads d_head, batch posn_k n_heads d_head -> batch n_heads posn_q posn_k"
) / (self.cfg.d_head ** 0.5)  # Scale by √d_head
```

**What's happening**: 
- For each position `i` and head `h`, compute dot product with all positions `j`
- `attn_scores[b, h, i, j]` = how much position `i` attends to position `j` in head `h`
- Scaling by `√d_head` prevents softmax from saturating

**Step 3: Causal Masking**
```python
# mask: [seq_len, seq_len] - lower triangular matrix
mask = torch.tril(torch.ones((seq_len, seq_len), device=residual.device))
# Set future positions to -inf (so softmax makes them 0)
attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
```

**What's happening**:
- Lower triangular matrix: `mask[i, j] = 1` if `j <= i`, else `0`
- Prevents token at position `i` from seeing tokens at positions `> i`
- Essential for autoregressive generation

**Step 4: Softmax and Apply to Values**
```python
# attn_pattern: [batch, n_heads, posn_q, posn_k] - probabilities
attn_pattern = torch.softmax(attn_scores, dim=-1)

# Weighted sum of values
# attn_output: [batch, posn_q, n_heads, d_head]
attn_output = einops.einsum(
    attn_pattern, v,
    "batch n_heads posn_q posn_k, batch posn_k n_heads d_head -> batch posn_q n_heads d_head"
)
```

**What's happening**:
- `attn_pattern[b, h, i, j]` = probability that position `i` attends to position `j`
- Weighted sum: `output[i] = Σ_j attn_pattern[i, j] * v[j]`

**Step 5: Project Back**
```python
# attn_output: [batch, posn, n_heads, d_head]
# W_O: [n_heads, d_head, d_model]
# output: [batch, posn, d_model]
output = einops.einsum(
    attn_output, self.W_O,
    "batch posn n_heads d_head, n_heads d_head d_model -> batch posn d_model"
)
```

**What's happening**: Combine all heads and project back to `d_model` dimensions.

#### `AttentionWithoutEinops`

Same logic, but using PyTorch operations:
```python
# Compute Q, K, V using einsum
q = torch.einsum("bpd,nhd->bpnh", residual, self.W_Q)

# Transpose for matmul: [batch, seq, n_heads, d_head] -> [batch, n_heads, seq, d_head]
q = q.transpose(1, 2)

# Attention scores
attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.cfg.d_head ** 0.5)
```

**Shape Flow (Full Attention)**:
```
Input: [batch, seq, d_model]
Q/K/V: [batch, seq, n_heads, d_head]  (per head)
Scores: [batch, n_heads, seq, seq]  (attention weights)
Pattern: [batch, n_heads, seq, seq]  (after softmax)
Output: [batch, seq, n_heads, d_head]  (weighted values)
Final: [batch, seq, d_model]  (after projection)
```

**Why Multi-Head?**
- Different heads can learn different relationships
- Head 1 might learn subject-verb relationships
- Head 2 might learn long-range dependencies
- Head 3 might learn local patterns
- Combining them gives richer representations

---

### 5. MLP / Feedforward Network (`pretraining/mlp/mlp.py`)

**Purpose**: Apply pointwise non-linear transformations to each position independently.

#### GPT Architecture (GELU)

```
Input → Linear(d_model → d_mlp) → GELU → Linear(d_mlp → d_model) → Output
```

Uses 2 weight matrices: `W_in` and `W_out`

#### LLaMA/OLMo Architecture (SwiGLU)

```
Input → [Gate Branch: Linear → Swish] × [Up Branch: Linear] → Linear(d_mlp → d_model) → Output
```

Uses 3 weight matrices: `W_gate`, `W_up`, and `W_out`

#### `MLPWithEinops`

```python
# First linear layer
# residual: [batch, posn, d_model]
# W_in: [d_model, d_mlp]
# hidden: [batch, posn, d_mlp]
hidden = einops.einsum(
    residual, self.W_in,
    "batch posn d_model, d_model d_mlp -> batch posn d_mlp"
) + self.b_in

# GELU activation (element-wise)
hidden = torch.nn.functional.gelu(hidden)

# Second linear layer
# hidden: [batch, posn, d_mlp]
# W_out: [d_mlp, d_model]
# output: [batch, posn, d_model]
output = einops.einsum(
    hidden, self.W_out,
    "batch posn d_mlp, d_mlp d_model -> batch posn d_model"
) + self.b_out
```

**What's happening**:
- Each position is processed independently (no interaction between positions)
- Expands to `d_mlp` (typically 4x `d_model`) for more capacity
- GELU provides non-linearity
- Projects back to `d_model`

**Why GELU?**
- GELU (Gaussian Error Linear Unit) is smoother than ReLU
- Used in GPT, BERT, and modern transformers
- Formula: `GELU(x) = x * Φ(x)` where Φ is CDF of standard normal

**SwiGLU (LLaMA/OLMo):**
- **Swish activation**: `Swish(x) = x * sigmoid(x)` (also called SiLU)
- **Gated architecture**: `SwiGLU(x) = Swish(W_gate @ x) * (W_up @ x)`
- Element-wise multiplication gates the information flow
- More expressive than GELU, allows model to control information flow
- Used in LLaMA, PaLM, OLMo, and other modern models

**Shape Flow (GELU):**
```
Input: [batch, posn, d_model]  # e.g., [32, 128, 256]
Expand: [batch, posn, d_mlp]  # e.g., [32, 128, 1024]
Activate: [batch, posn, d_mlp]  # GELU (element-wise)
Project: [batch, posn, d_model]  # e.g., [32, 128, 256]
```

**Shape Flow (SwiGLU):**
```
Input: [batch, posn, d_model]  # e.g., [32, 128, 256]
Gate: [batch, posn, d_mlp]  # Swish(W_gate @ x)
Up: [batch, posn, d_mlp]  # W_up @ x
Hidden: [batch, posn, d_mlp]  # gate * up (element-wise)
Project: [batch, posn, d_model]  # e.g., [32, 128, 256]
```

---

### 6. Transformer Block (`pretraining/transformer_blocks/transformer_block.py`)

**Purpose**: Combine attention and MLP with residual connections and layer normalization.

#### Architecture (Pre-Norm)

```
Input
  ↓
LayerNorm → Attention → + (residual)
  ↓                      ↑
  └──────────────────────┘
  ↓
LayerNorm → MLP → + (residual)
  ↓                ↑
  └────────────────┘
  ↓
Output
```

#### Implementation

```python
def forward(self, residual):
    # Pre-norm attention with residual connection
    residual = residual + self.attn(self.ln1(residual))
    
    # Pre-norm MLP with residual connection
    residual = residual + self.mlp(self.ln2(residual))
    
    return residual
```

**Key Concepts**:

1. **Pre-Norm vs Post-Norm**:
   - **Pre-Norm** (what we use): `x + f(LN(x))`
   - **Post-Norm**: `LN(x + f(x))`
   - Pre-norm is more stable for deep networks

2. **Residual Connections**:
   - Allow gradients to flow directly through
   - Enable training of very deep networks
   - Help model learn identity function if needed

3. **Why Two LayerNorms?**
   - One before attention (stabilizes attention)
   - One before MLP (stabilizes MLP)
   - Each sub-block gets normalized inputs

**Shape Flow:**
```
Input: [batch, posn, d_model]
  ↓
LN1: [batch, posn, d_model]
  ↓
Attention: [batch, posn, d_model]
  ↓
Add: [batch, posn, d_model]  (residual connection)
  ↓
LN2: [batch, posn, d_model]
  ↓
MLP: [batch, posn, d_model]
  ↓
Add: [batch, posn, d_model]  (residual connection)
  ↓
Output: [batch, posn, d_model]
```

---

### 7. Full Transformer Model (`pretraining/model/model.py`)

**Purpose**: Stack all components into a complete language model supporting GPT, LLaMA, and OLMo architectures.

#### Architecture Flow

**GPT Architecture:**
```
Tokens [batch, position]
  ↓
Token Embeddings → [batch, position, d_model]
  ↓
+ Learned Positional Embeddings → [batch, position, d_model]
  ↓
Transformer Block 1 → [batch, position, d_model]
  ↓
...
  ↓
Transformer Block N → [batch, position, d_model]
  ↓
Final LayerNorm → [batch, position, d_model]
  ↓
Unembedding → [batch, position, d_vocab] (logits)
```

**LLaMA Architecture:**
```
Tokens [batch, position]
  ↓
Token Embeddings → [batch, position, d_model]
  ↓
(No positional embedding layer - RoPE applied in attention)
  ↓
Transformer Block 1 (with RoPE) → [batch, position, d_model]
  ↓
...
  ↓
Transformer Block N (with RoPE) → [batch, position, d_model]
  ↓
Final RMSNorm → [batch, position, d_model]
  ↓
Unembedding → [batch, position, d_vocab] (logits)
```

**OLMo Architecture:**
```
Tokens [batch, position]
  ↓
Token Embeddings → [batch, position, d_model]
  ↓
(No positional embedding layer - ALiBi applied in attention)
  ↓
Transformer Block 1 (with ALiBi) → [batch, position, d_model]
  ↓
...
  ↓
Transformer Block N (with ALiBi) → [batch, position, d_model]
  ↓
Final LayerNorm → [batch, position, d_model]
  ↓
Unembedding → [batch, position, d_vocab] (logits)
```

#### Implementation

The model automatically selects components based on `cfg.architecture`:

```python
def forward(self, tokens):
    # tokens: [batch, position]
    
    # Token embeddings
    residual = self.embed(tokens)  # [batch, position, d_model]
    
    # Positional embeddings (GPT only)
    if self.pos_embed is not None:  # GPT
        residual = residual + self.pos_embed(tokens)
    # LLaMA: RoPE is applied inside attention blocks
    # OLMo: ALiBi is applied inside attention blocks
    
    # Pass through transformer blocks
    for block in self.blocks:
        residual = block(residual)  # [batch, position, d_model]
    
    # Final normalization (LayerNorm for GPT/OLMo, RMSNorm for LLaMA)
    residual = self.ln_f(residual)  # [batch, position, d_model]
    
    # Unembedding to logits
    logits = torch.matmul(residual, self.unembed)
    
    return logits
```

---

## Resources

### Papers

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) - Original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971) - LLaMA paper
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/pdf/2402.00838) - OLMo paper

### Transformers

- [ARENA's Transformers from Scratch](https://arena-chapter1-transformer-interp.streamlit.app/%5B1.1%5D_Transformer_from_Scratch)
- [Neel Nanda on building an LLM from scratch](https://www.youtube.com/watch?v=bOYE6E8JrtU&list=PL7m7hLIqA0hoIUPhC26ASCVs_VrqcDpAz)
- [Andrej Karpathy on building an LLM from scratch](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [3Blue1Brown on LLMs](https://www.youtube.com/watch?v=LPZh9BOjkQs&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=6)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [John Hewitt at Stanford on pre-training](https://www.youtube.com/watch?v=DGfCRXuNA2w&list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D&index=11)

### Einops
- [Einops Documentation](https://einops.rocks/) - Learn more about einops