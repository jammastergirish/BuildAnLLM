# Build an LLM

![ezgif-59c817502b85280b](https://github.com/user-attachments/assets/3437fd07-b7bd-4497-ac7e-0494ea8d00f3)

This repository contains an educational training workflow for a transformer-based autoregressive, decoder-only language model. It is optimized not for speed or cost, but rather for learning.

Users can:

- **Pre-train an LLM from scratch** using a simple, intuitive interface, with a diagram that visualizes the user's specific configuration, as well as attention heatmaps.
- **Fine-tune a pre-trained model** on prompt/response pairs using supervised fine-tuning (SFT), with support for both full-parameter and parameter-efficient fine-tuning via LoRA.
- **Explore the code** to understand the modularized building blocks of transformer models, with multiple implementation variants for each component. The code shown dynamically adapts to configuration choices.
- **Work through equations** to understand the math behind the code, with equations dynamically displayed based on configuration.

I built this as I wanted to properly understand LLMs. A great way to learn is to write code yourself; an even better way to learn is to write code in a general, modular manner that's clean enough for others to read.

I'm incredibly grateful to all those from whom I learned and borrowed ideas (see [Resources](#resources)). I hope others find this repository helpful!

_(Comments, corrections, and pull requests are very welcome. Currently, this repository serves a decoder-only architecture (like GPT, LLaMA, OLMo) and includes Mixture of Experts (MoE) architectures (like DeepSeek-V2 and Mixtral). It does not include encoder-only models (like BERT) or encoder-decoder models (like T5). It includes pre-training and supervised fine-tuning (SFT), but not yet RLHF or other advanced fine-tuning techniques.)_

## Quickstart ##

```bash
uv run --with streamlit streamlit run main.py
```

## What You'll Learn

By exploring the interface and codebase, you'll gain a deep understanding of:

- **Transformer Architecture**: How decoder-only language models work from the ground up
- **Core Components**: Attention mechanisms, normalization layers, positional encodings, and feedforward networks
- **Architecture Variants**: Differences between GPT, LLaMA, and OLMo implementations
- **Mixture of Experts (MoE)**: How MoE architectures use multiple expert MLPs with routing to scale model capacity efficiently
- **Pre-Training Process**: How to pre-train a language model using next-token prediction on raw text
- **Fine-Tuning Process**: How to fine-tune a pre-trained model on prompt/response pairs using supervised fine-tuning (SFT)
- **Loss Masking**: How to compute loss only on response tokens (not prompt tokens) during fine-tuning
- **Text Generation**: Autoregressive generation and sampling strategies (temperature, top-k, top-p)
- **KV Caching**: Efficient inference optimization that speeds up text generation by caching key-value tensors
- **Implementation Details**: Multiple implementation approaches (with/without einops, manual vs PyTorch built-ins) to understand what's happening under the hood

The codebase includes both educational implementations (showing the math and operations explicitly) and optimized versions, so you can see how concepts translate to actual code.

## Contents

1. [Usage Guide](#usage-guide)
2. [Key Concepts](#key-concepts)
3. [Pre-Training Pipeline](#pre-training-pipeline)
4. [Fine-Tuning Pipeline](#fine-tuning-pipeline)
5. [Inference and Sampling](#inference-and-sampling)
6. [Core Components Deep Dive](#core-components-deep-dive)
7. [Resources](#resources)

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
- **Fine-Tuning Page**: Fine-tune pre-trained models on prompt/response pairs
- **Inference Page**: Generate text from trained models (pre-trained or fine-tuned)

### Pre-Training

1. Loads training text file
2. Creates tokenizer and dataset
3. Initializes model based on selected architecture and configuration
4. Trains for specified epochs with real-time loss visualization
5. Saves checkpoints to `checkpoints/YYYYMMDDHHMMSS/` (timestamped folders)

#### UI

1. Upload training data or use the default `training.txt` file
2. Select an architecture preset, or choose custom parameters:
   - **🚀 GPT-2**: Learned positional embeddings, LayerNorm, GELU activation
   - **🦙 LLaMA**: RoPE positional encoding, RMSNorm, SwiGLU activation
   - **🔬 OLMo**: ALiBi positional encoding, LayerNorm, SwiGLU activation
   - **🔷 DeepSeek V2**: LLaMA-style with MoE (64 experts, top-6, 2 shared experts)
   - **🎯 Mixtral**: LLaMA-style with MoE (8 experts, top-2 routing). Sparse MoE architecture matching Mixtral 8x7B design
3. Configure model dimensions (or use size presets: small, medium, full)
4. Optionally enable MoE (Mixture of Experts) and configure expert settings
5. Set training hyperparameters (batch size, learning rate, epochs, etc.)
6. Click **"Start Training"** to enter the **Interactive Training Mode**:
   - **Glass Box Training**: Watch the model learn step-by-step.
   - **Colored Tokens**: See exactly how the tokenizer splits your input text with color-coded tokens (hover for IDs).
   - **Real-time Controls**: **Start/Pause/Resume** and ⏭️ **Step to Next Batch** buttons for manual stepping.
   - **Attention Heatmaps**: Visualize how the model attends to different tokens in the sequence (visible when paused).
   - **Live Metrics**: Monitor Loss, Gradient Norms, and Validation performance in real-time.

#### Command-line

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

**Options**:
- `--checkpoint`: Path to model checkpoint (e.g., `checkpoints/20240101120000/final_model.pt`)
- `--prompt`: Starting text prompt
- `--max_new_tokens`: Number of tokens to generate (default: 200)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top_k`: Top-k sampling (optional)
- `--top_p`: Top-p sampling (optional, default: 0.9)
- `--tokenizer_type`: Tokenizer type (optional, auto-detected from checkpoint; only needed for old checkpoints)
- `--text_file`: Text file for character tokenizer initialization (default: `training.txt`, only needed for character tokenizer)

### UI

1. Select a checkpoint from the dropdown (auto-scans `checkpoints/` directory)
   - Shows both pre-trained and fine-tuned checkpoints
   - Clearly labeled: "🏁 Final Model (Pre-trained)" vs "🏁 Final Model (Fine-tuned)"
   - Fine-tuned models are in `checkpoints/{timestamp}/sft/` subdirectories
2. Enter a prompt
3. Configure sampling parameters (temperature, top-k, top-p)
4. Click "Generate" to create text
5. **Glass Box Internals**: Scroll down to inspect the model's internal state during generation:
   - **Attention Maps**: Visualize where the model is looking.
   - **Logit Lens**: See what the model "thinks" the next token is after every layer.
   - **Layer Norms**: Track signal propagation through the network.

#### Command-line

```bash
# Generate text from trained model
uv run cli/infer.py --checkpoint checkpoints/20240101120000/final_model.pt --prompt "First Citizen:"
```

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

### 3. Attention Mechanisms & Heatmaps

**What it is**: The core mechanism that allows the model to "look back" at previous tokens to inform the current prediction.

**The Intuition**:
Imagine reading a sentence. When you see the word "bank", you need to know if the context is "river" or "money" to understand it. Attention allows the model to look back at words like "river" or "money" earlier in the sentence to disambiguate "bank".

**Heatmap Visualization**:
In the Glass Box view (Pre-Training and Inference), you can see this process happening via **Attention Heatmaps**:
- **X-axis (Key)**: The tokens the model is looking *at* (source).
- **Y-axis (Query)**: The token the model is currently *generating* (destination).
- **Color Intensity**: Darker/Brighter colors indicate stronger focus.
- **Diagonal Pattern**: A strong diagonal line means the model is mostly looking at the immediate previous token (common in early layers).

![Attention Heatmap Example](assets/attention_heatmap.png)

### 4. Residual Connections

**What it is**: Adding input to output: `output = input + transformation(input)`

**Why it helps**:
- Allows gradients to flow directly through
- Enables training of very deep networks
- The model can learn the identity function if the transformation isn't needed

### 5. Layer Normalization

**What it is**: Normalizing activations across the feature dimension.

**Why it helps**:
- Stabilizes training
- Allows higher learning rates
- Reduces internal covariate shift

**Variants**:
- **LayerNorm** (GPT/OLMo): Normalizes by subtracting mean, then scaling
- **RMSNorm** (LLaMA): Only scales (no mean subtraction, no bias)

### 6. Positional Encoding

**The Problem**: Transformers have no inherent notion of sequence order.

**Solutions**:
- **Learned Embeddings** (GPT): Fixed embeddings for each position
- **RoPE** (LLaMA): Rotates query/key vectors by position-dependent angles
- **ALiBi** (OLMo): Adds distance-based bias to attention scores

### 7. Pre-training vs Fine-tuning

**Pre-training**:
- Train on large, diverse text corpus
- Learn general language patterns
- Unsupervised (no labels needed)
- Example: Train on Wikipedia, books, web text

**Supervised Fine-Tuning (SFT)**:
- Take pre-trained model
- Train further on prompt/response pairs
- Supervised (needs labeled data: prompt → response)
- Lower learning rate (typically 10-100x lower)
- Shorter training (1-5 epochs vs 10+)
- **Loss masking**: Only compute loss on response tokens, not prompt tokens
- Example: Train on instruction-following datasets

**Why Fine-Tune?**
- Pre-trained models learn general language but may not follow instructions well
- Fine-tuning teaches the model to respond appropriately to prompts
- Makes the model more useful for specific tasks (Q&A, instruction following, etc.)

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
- **MoE Configuration** (when `use_moe=True`):
  - `num_experts`: Number of expert MLPs (e.g., 8, 64)
  - `num_experts_per_tok`: Top-k experts to activate per token (e.g., 2, 6)
  - `use_shared_experts`: Enable shared experts (DeepSeek-style)
  - `num_shared_experts`: Number of always-active shared experts
  - `router_type`: Routing strategy (`top_k` or `top_k_with_shared`)
  - `load_balancing_loss_weight`: Weight for load balancing auxiliary loss
  - `expert_capacity_factor`: Capacity factor for expert load balancing

#### `TransformerTrainingArgs` - Training Hyperparameters
- `batch_size`: Number of sequences per batch
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `weight_decay`: L2 regularization strength
- `eval_iters`: Number of batches for evaluation

---

## Fine-Tuning Pipeline

### 1. Data Loading (`finetuning/data/sft_dataset.py`)

**Purpose**: Load prompt/response pairs from CSV and create training sequences with loss masks.

#### Process

1. **Load CSV**: Read CSV file with `prompt` and `response` columns
2. **Tokenize**: Convert prompts and responses to token IDs
3. **Create Sequences**: Concatenate prompt + response
4. **Create Masks**: 0 for prompt tokens, 1 for response tokens
5. **Shift by 1**: Create input/target pairs (same as pre-training)
6. **Split**: Train/validation split (default 90/10)

#### Code Flow

```python
# Load CSV
df = pd.read_csv("finetuning.csv")
prompts = df['prompt'].tolist()
responses = df['response'].tolist()

# For each pair
for prompt, response in zip(prompts, responses):
    # Tokenize
    prompt_tokens = tokenizer.encode(prompt)  # [prompt_len]
    response_tokens = tokenizer.encode(response)  # [response_len]
    
    # Concatenate
    full_tokens = prompt_tokens + response_tokens  # [prompt_len + response_len]
    
    # Create mask: 0 for prompt, 1 for response
    mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)
    
    # Shift by 1 (same as pre-training)
    input_seq = full_tokens[:-1]  # [seq_len-1]
    target_seq = full_tokens[1:]   # [seq_len-1]
    mask_seq = mask[1:]            # [seq_len-1]

# X: [num_sequences, seq_len-1] - input sequences
# Y: [num_sequences, seq_len-1] - target sequences
# masks: [num_sequences, seq_len-1] - loss masks (1 for response, 0 for prompt)
```

**Why Mask Prompt Tokens?**
- We want the model to learn to generate responses, not repeat prompts
- Computing loss on prompt tokens would teach the model to copy the prompt
- Masking ensures we only learn from the response tokens

### 2. Training Loop (`finetuning/training/sft_trainer.py`)

**Purpose**: Fine-tune the pre-trained model using masked loss.

#### Key Differences from Pre-Training

1. **Masked Loss**: Only compute loss on response tokens
2. **Lower Learning Rate**: Typically 1e-5 (vs 1e-3 for pre-training)
3. **Shorter Training**: 1-5 epochs (vs 10+ for pre-training)
4. **Structured Data**: Prompt/response pairs instead of raw text

#### Training Step

```python
# 1. Get random batch
idx = torch.randint(0, len(X_train), (batch_size,))
x_batch = X_train[idx]  # [batch_size, seq_len]
y_batch = Y_train[idx]   # [batch_size, seq_len]
masks_batch = masks_train[idx]  # [batch_size, seq_len]

# 2. Forward pass
logits = model(x_batch)  # [batch_size, seq_len, vocab_size]

# 3. Compute masked loss
logits_flat = logits.view(-1, vocab_size)  # [batch*seq, vocab]
targets_flat = y_batch.view(-1)            # [batch*seq]
masks_flat = masks_batch.view(-1)          # [batch*seq]

# Compute loss per token
loss_unmasked = F.cross_entropy(
    logits_flat, targets_flat, reduction='none'
)  # [batch*seq]

# Apply mask: only average over response tokens (where mask == 1)
loss = (loss_unmasked * masks_flat).sum() / masks_flat.sum().clamp(min=1)

# 4. Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**Loss Explanation**:
- `loss_unmasked`: Loss for every token position
- `masks_flat`: 1 for response tokens, 0 for prompt tokens
- `loss_unmasked * masks_flat`: Zero out prompt token losses
- `sum() / masks_flat.sum()`: Average only over response tokens

**Why Lower Learning Rate?**
- Pre-trained weights are already good
- We want small adjustments, not large changes
- Prevents catastrophic forgetting of pre-trained knowledge

### 3. Checkpoint Organization

**Pre-trained checkpoints**: `checkpoints/{timestamp}/final_model.pt`
**Fine-tuned checkpoints**: `checkpoints/{timestamp}/sft/final_model.pt`

Both checkpoints are visible in the inference page, clearly labeled:
- "🏁 Final Model (Pre-trained)"
- "🏁 Final Model (Fine-tuned)"

---

## Inference and Sampling

### Text Generation (`inference/sampler.py`)

**Note**: The inference page supports both pre-trained and fine-tuned models. Fine-tuned models are often better at following instructions and generating appropriate responses to prompts.

**Purpose**: Generate text from a trained model.

#### Autoregressive Generation

**Basic Approach (without KV cache)**:
```python
# Start with prompt
tokens = tokenizer.encode(prompt)  # [seq_len]

# Generate tokens one by one
for _ in range(max_new_tokens):
    # Get model predictions (recomputes everything each time)
    logits = model(tokens)  # [1, seq_len, vocab_size]
    
    # Get logits for last position
    next_token_logits = logits[0, -1, :]  # [vocab_size]
    
    # Sample next token
    next_token = sample(next_token_logits)
    
    # Append to sequence
    tokens.append(next_token)
```

**Optimized Approach (with KV cache)**:
```python
# Start with prompt
tokens = tokenizer.encode(prompt)  # [seq_len]
tokens_tensor = torch.tensor([tokens], device=device)

# Process prompt and initialize KV cache
logits, kv_cache = model(tokens_tensor, cache=None, start_pos=0)
next_token_logits = logits[0, -1, :] / temperature
start_pos = tokens_tensor.shape[1]

# Generate tokens one by one
for _ in range(max_new_tokens):
    # Sample next token
    next_token = sample(next_token_logits)
    
    # Append to sequence
    tokens_tensor = torch.cat([tokens_tensor, next_token.unsqueeze(0)], dim=1)
    
    # Process only the new token using KV cache
    # This is much faster - only computes Q, K, V for the new token
    new_token_tensor = next_token.unsqueeze(0)  # [1, 1]
    logits, kv_cache = model(new_token_tensor, cache=kv_cache, start_pos=start_pos)
    next_token_logits = logits[0, -1, :] / temperature
    start_pos += 1
```

**Why KV Cache?**
- **Without cache**: Each generation step recomputes Q, K, V for all previous tokens (O(n²) complexity per step)
- **With cache**: Each generation step only computes Q, K, V for the new token, reusing cached K, V from previous tokens (O(n) complexity per step)
- **Speedup**: Dramatically faster for long sequences - can be 10-100x faster depending on sequence length
- **Memory tradeoff**: Uses more memory to store cached K, V tensors, but the speedup is usually worth it

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

**Unified Implementation:**

The `LayerNorm` class supports both einops and PyTorch implementations via the `use_einops` flag:

#### With Einops (`use_einops=True`)
Uses `einops.reduce` for mean and variance computation:
```python
# Compute mean: [batch, posn, d_model] -> [batch, posn, 1]
residual_mean = einops.reduce(
    residual, 'batch posn d_model -> batch posn 1', 'mean'
)
```

**Why this works**: Einops makes it explicit that we're reducing over `d_model` while keeping `batch` and `posn`.

#### Without Einops (`use_einops=False`)
Uses PyTorch's built-in operations:
```python
# Compute mean: [batch, posn, d_model] -> [batch, posn, 1]
residual_mean = residual.mean(dim=-1, keepdim=True)
```

**Why this works**: `dim=-1` means "last dimension" (d_model), `keepdim=True` preserves the dimension.

#### PyTorch Built-in (`LayerNormWithTorch`)
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

The `PosEmbed` class supports both einops and PyTorch implementations via the `use_einops` flag:

#### With Einops (`use_einops=True`)
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

#### Without Einops (`use_einops=False`)
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

#### Attention Types: MHA, GQA, and MQA

The codebase supports three attention variants:

**1. Multi-Head Attention (MHA)** - Standard attention:
- Each head has separate Q, K, V projections
- `n_kv_heads = n_heads` (default)
- Used in: GPT-2, original LLaMA
- KV cache size: `[batch, seq_len, n_heads, d_head]`

**2. Grouped Query Attention (GQA)** - Efficient attention:
- Groups of Q heads share the same K/V heads
- `n_kv_heads < n_heads` (e.g., 32 Q heads, 8 KV heads = 4:1 ratio)
- Used in: LLaMA 2, Mistral, Mixtral
- KV cache size: `[batch, seq_len, n_kv_heads, d_head]` (smaller!)
- Benefits: ~75% smaller KV cache, faster inference, minimal quality loss

**3. Multi-Query Attention (MQA)** - Most efficient:
- All Q heads share a single K/V head
- `n_kv_heads = 1`
- Used in: PaLM, some optimized models
- KV cache size: `[batch, seq_len, 1, d_head]` (much smaller!)
- Benefits: Maximum memory efficiency, faster inference, slight quality trade-off

**How GQA/MQA Works**:
1. Compute Q with `n_heads` projections: `Q: [batch, seq, n_heads, d_head]`
2. Compute K/V with `n_kv_heads` projections: `K, V: [batch, seq, n_kv_heads, d_head]`
3. Broadcast K/V to match Q: repeat each KV head `n_heads / n_kv_heads` times
4. Attention computation proceeds identically to MHA after broadcasting
5. Cache stores original (non-broadcasted) K/V to save memory

#### `Attention` - Unified Implementation

The `Attention` class supports both einops and PyTorch implementations via the `use_einops` flag:

**With Einops (`use_einops=True`):**

**Step 1: Compute Q, K, V**
```python
# residual: [batch, posn, d_model]
# W_Q: [n_heads, d_head, d_model]
# W_K, W_V: [n_kv_heads, d_head, d_model] (for GQA/MQA)
# q: [batch, posn, n_heads, d_head]
q = einops.einsum(
    residual, self.W_Q,
    "batch posn d_model, n_heads d_head d_model -> batch posn n_heads d_head"
)
# k: [batch, posn, n_kv_heads, d_head] (may be different from n_heads)
k = einops.einsum(
    residual, self.W_K,
    "batch posn d_model, n_kv_heads d_head d_model -> batch posn n_kv_heads d_head"
)
# v: [batch, posn, n_kv_heads, d_head]
v = einops.einsum(
    residual, self.W_V,
    "batch posn d_model, n_kv_heads d_head d_model -> batch posn n_kv_heads d_head"
)
```

**What's happening**: 
- Q is projected with `n_heads` projections (one per Q head)
- K/V are projected with `n_kv_heads` projections (may be fewer than `n_heads` for GQA/MQA)
- For MHA: `n_kv_heads = n_heads` (standard behavior)
- For GQA/MQA: `n_kv_heads < n_heads` (memory efficient)

**Step 1b: Broadcast K/V for GQA/MQA** (if `n_kv_heads < n_heads`)
```python
# Broadcast K/V to match Q heads
if self.n_kv_heads < self.n_heads:
    repeat_factor = self.n_heads // self.n_kv_heads
    k = k.repeat_interleave(repeat_factor, dim=2)  # [batch, posn_k, n_heads, d_head]
    v = v.repeat_interleave(repeat_factor, dim=2)  # [batch, posn_k, n_heads, d_head]
```

**Step 2: Compute Attention Scores**
```python
# After broadcasting, k and v have n_heads dimension
# q: [batch, posn_q, n_heads, d_head]
# k: [batch, posn_k, n_heads, d_head] (broadcasted if GQA/MQA)
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

**Without Einops (`use_einops=False`):**

Same logic, but using PyTorch operations:
```python
# Compute Q, K, V using einsum
q = torch.einsum("bpd,nhd->bpnh", residual, self.W_Q)  # [batch, seq, n_heads, d_head]
k = torch.einsum("bpd,nkd->bpnk", residual, self.W_K)  # [batch, seq, n_kv_heads, d_head]
v = torch.einsum("bpd,nkd->bpnk", residual, self.W_V)  # [batch, seq, n_kv_heads, d_head]

# Broadcast K/V for GQA/MQA (if needed)
if self.n_kv_heads < self.n_heads:
    repeat_factor = self.n_heads // self.n_kv_heads
    k = k.repeat_interleave(repeat_factor, dim=2)  # [batch, seq, n_heads, d_head]
    v = v.repeat_interleave(repeat_factor, dim=2)  # [batch, seq, n_heads, d_head]

# Transpose for matmul: [batch, seq, n_heads, d_head] -> [batch, n_heads, seq, d_head]
q = q.transpose(1, 2)
k = k.transpose(1, 2)
v = v.transpose(1, 2)

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

#### KV Caching for Efficient Inference

**The Problem**: During autoregressive generation, we generate tokens one at a time. Without caching, each step recomputes Q, K, V for all previous tokens, which is wasteful since K and V for previous tokens don't change.

**The Solution**: Cache K and V tensors from previous tokens, and only compute Q, K, V for the new token.

**How it works**:
```python
# First forward pass (processing prompt)
# tokens: [batch, prompt_len]
logits, kv_cache = model(tokens, cache=None, start_pos=0)
# kv_cache: List of (K_cache, V_cache) tuples, one per layer
# For MHA: K_cache, V_cache: [batch, prompt_len, n_heads, d_head]
# For GQA/MQA: K_cache, V_cache: [batch, prompt_len, n_kv_heads, d_head] (smaller!)

# Subsequent forward passes (generating new tokens)
# new_token: [batch, 1] - only the new token
logits, kv_cache = model(new_token, cache=kv_cache, start_pos=prompt_len)
# K, V are concatenated: [cached_K, new_K] and [cached_V, new_V]
# Only new_K and new_V are computed, cached ones are reused
# For GQA/MQA: Cache stores original (non-broadcasted) K/V to save memory
```

**Implementation Details**:
1. **Attention Layer**: Accepts optional `cache` parameter containing cached K, V from previous tokens
2. **Concatenation**: New K, V are concatenated with cached K, V along sequence dimension
3. **RoPE Handling**: Cached K already has RoPE applied, so only new K gets rotated
4. **ALiBi Handling**: Bias matrix is computed for full sequence length (cached + new tokens)
5. **Cache Update**: Returns updated cache containing K, V for all tokens (cached + new)

**Benefits**:
- **Speed**: 10-100x faster for long sequences
- **Efficiency**: Only computes Q, K, V for new tokens (O(n) instead of O(n²) per step)
- **Memory**: Trades memory for speed (stores cached K, V tensors)

**Backward Compatibility**: When `cache=None`, the model works exactly as before (used during training).

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

#### `MLP` - Unified Implementation

The `MLP` class supports both einops and PyTorch implementations via the `use_einops` flag:

**With Einops (`use_einops=True`):**

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

**Without Einops (`use_einops=False`):**

Same logic using PyTorch operations (typically using `torch.einsum` or `torch.matmul`).

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

#### Attention Types: MHA, GQA, and MQA

**Multi-Head Attention (MHA)**:
- Standard attention: each head has separate Q, K, V
- KV cache: `[batch, seq_len, n_heads, d_head]`
- Used in: GPT-2, original LLaMA

**Grouped Query Attention (GQA)**:
- Groups of Q heads share K/V heads (e.g., 32 Q heads, 8 KV heads)
- KV cache: `[batch, seq_len, n_kv_heads, d_head]` (75% smaller for 4:1 ratio!)
- Used in: LLaMA 2, Mistral, Mixtral
- Benefits: Smaller KV cache, faster inference, minimal quality loss

**Multi-Query Attention (MQA)**:
- All Q heads share single K/V head
- KV cache: `[batch, seq_len, 1, d_head]` (much smaller!)
- Used in: PaLM, optimized models
- Benefits: Maximum memory efficiency, faster inference

**Implementation**: K/V are computed with `n_kv_heads` projections, then broadcast to match Q heads before attention computation. Cache stores original (non-broadcasted) K/V.

---

#### MoE Architecture (Mixture of Experts)

**Purpose**: Scale model capacity efficiently by using multiple expert MLPs and routing tokens to a subset of experts.

**Key Concept**: Instead of one large MLP, MoE uses multiple smaller expert MLPs. For each token, a router selects the top-k experts to activate, allowing the model to have more parameters while keeping computation per token similar.

**Architecture Flow:**
```
Input: [batch, posn, d_model]
  ↓
Router Network → [batch, posn, num_experts] (logits)
  ↓
Softmax → Router Probabilities
  ↓
Top-k Selection → Select k experts per token
  ↓
Expert MLPs (only selected experts compute)
  ↓
Weighted Combination → [batch, posn, d_model]
  ↓
(+ Shared Experts if enabled)
  ↓
Output: [batch, posn, d_model]
```

**Routing Strategies:**

1. **Top-k Routing** (Mixtral-style):
   - Router computes logits for all experts
   - Selects top-k experts per token
   - Combines expert outputs with routing weights
   - Only k experts compute per token (sparse activation)

2. **Top-k with Shared Experts** (DeepSeek-style):
   - Same as top-k, but some experts are always active
   - Shared experts handle general knowledge
   - Routed experts specialize based on input
   - Final output = shared_experts(x) + routed_experts(x)

**Load Balancing Loss:**
- Auxiliary loss encourages uniform expert usage
- Prevents expert collapse (where only a few experts are used)
- Formula: `aux_loss = num_experts * sum(P_i * f_i)` where:
  - `P_i` = average routing probability for expert i
  - `f_i` = fraction of tokens routed to expert i
- Added to main loss: `total_loss = loss + aux_loss * load_balancing_loss_weight`

**Benefits:**
- **Scalability**: Can have many experts (e.g., 64) while only activating a few per token
- **Efficiency**: Computation scales with activated experts, not total experts
- **Specialization**: Different experts can specialize in different patterns

**Implementation** (`pretraining/mlp/mlp.py`):

```python
class MoEMLPBase(nn.Module):
    def forward(self, residual):
        # Router: [batch, seq_len, d_model] -> [batch, seq_len, num_experts]
        router_logits = self.router(residual)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(
            router_probs, k=self.num_experts_per_tok, dim=-1
        )
        
        # Process each expert
        output = torch.zeros_like(residual)
        for expert_idx in range(self.num_experts):
            # Get expert output and weight by routing probability
            expert_output = self.experts[expert_idx](residual)
            expert_weights = ...  # Extract from top_k_probs
            output += expert_weights.unsqueeze(-1) * expert_output
        
        # Add shared experts if enabled
        if self.use_shared_experts:
            shared_output = sum(expert(residual) for expert in self.shared_experts)
            output += shared_output / len(self.shared_experts)
        
        # Compute load balancing loss
        aux_loss = self._compute_load_balancing_loss(...)
        
        return output, aux_loss
```

**Shape Flow:**
```
Input: [batch, posn, d_model]  # e.g., [32, 128, 256]
Router: [batch, posn, num_experts]  # e.g., [32, 128, 8]
Top-k: [batch, posn, k]  # e.g., [32, 128, 2] (indices and probs)
Expert Outputs: [batch, posn, d_model]  # from each selected expert
Weighted Sum: [batch, posn, d_model]  # e.g., [32, 128, 256]
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

**MoE Architecture (DeepSeek V2 / Mixtral-style):**
```
Tokens [batch, position]
  ↓
Token Embeddings → [batch, position, d_model]
  ↓
(No positional embedding layer - RoPE applied in attention)
  ↓
Transformer Block 1 (with RoPE + MoE MLP) → [batch, position, d_model]
  ↓
...
  ↓
Transformer Block N (with RoPE + MoE MLP) → [batch, position, d_model]
  ↓
Final RMSNorm → [batch, position, d_model]
  ↓
Unembedding → [batch, position, d_vocab] (logits)
```

**MoE Transformer Block:**
```
Input: [batch, posn, d_model]
  ↓
LayerNorm → Attention → + (residual)
  ↓
LayerNorm → MoE MLP → + (residual)
  │         │
  │         ├─ Router → Top-k Selection
  │         ├─ Expert MLPs (sparse activation)
  │         └─ Load Balancing Loss (auxiliary)
  ↓
Output: [batch, posn, d_model]
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
    aux_losses = []
    for block in self.blocks:
        residual, aux_loss = block(residual)  # [batch, position, d_model]
        if aux_loss is not None:  # MoE auxiliary loss
            aux_losses.append(aux_loss)
    
    # Aggregate MoE auxiliary losses
    total_aux_loss = sum(aux_losses) if aux_losses else None
    
    # Final normalization (LayerNorm for GPT/OLMo, RMSNorm for LLaMA)
    residual = self.ln_f(residual)  # [batch, position, d_model]
    
    # Unembedding to logits
    logits = torch.matmul(residual, self.unembed)
    
    # Return logits and auxiliary loss if MoE is enabled
    if total_aux_loss is not None:
        return logits, total_aux_loss
    return logits
```

---

## Resources

### LLM from Scratch

- [ARENA's Transformers from Scratch](https://arena-chapter1-transformer-interp.streamlit.app/%5B1.1%5D_Transformer_from_Scratch)
- [Neel Nanda on building an LLM from scratch](https://www.youtube.com/watch?v=bOYE6E8JrtU&list=PL7m7hLIqA0hoIUPhC26ASCVs_VrqcDpAz)
- [Andrej Karpathy on building an LLM from scratch](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Sebastian Raschka's Build an LLM from Scratch](https://sebastianraschka.com/llms-from-scratch/)

### Background

- [3Blue1Brown on LLMs](https://www.youtube.com/watch?v=LPZh9BOjkQs&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=6)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [John Hewitt at Stanford on pre-training](https://www.youtube.com/watch?v=DGfCRXuNA2w&list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D&index=11)
- [Tom Yeh's AI By Hand](https://www.byhand.ai/)

### Architectures

- [Sebastian Raschka's Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)

### Papers

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) - Original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971) - LLaMA paper
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/pdf/2402.00838) - OLMo paper
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/pdf/1701.06538) - Original MoE paper
- [Mixtral of Experts](https://arxiv.org/pdf/2401.04088) - Mixtral MoE paper
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/pdf/2405.04434) - DeepSeek V2 MoE paper

### Documentation
- [PyTorch](https://docs.pytorch.org/docs/stable/index.html)
- [Einops](https://einops.rocks/)
- [Streamlit](https://docs.streamlit.io/)
