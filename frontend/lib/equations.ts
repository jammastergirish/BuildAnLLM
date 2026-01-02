export const modelEquations = String.raw`
### Key notation
- **x**: input tensor [B, L, d_model]
- **W_Q, W_K, W_V, W_O**: attention matrices
- **W_in, W_out**: MLP matrices
- **d_model**: model dimension
- **n_heads**: number of attention heads

---
### Token embedding
$$E \in \mathbb{R}^{V \times d_{model}}$$

$$x_0 = E[\text{tokens}]$$

---
### Positional encoding
If learned positions:
$$x_0 = x_0 + P[\text{positions}]$$

---
### Attention
$$Q = x W_Q, \; K = x W_K, \; V = x W_V$$

$$\text{attn}(x) = \text{softmax}(QK^T / \sqrt{d_{head}}) V$$

---
### MLP
$$\text{mlp}(x) = W_{out} \sigma(W_{in} x)$$

---
### Residual block
$$x_{i+1} = x_i + \text{attn}(x_i)$$

$$x_{i+2} = x_{i+1} + \text{mlp}(x_{i+1})$$
`;

export const inferenceEquations = String.raw`
### Autoregressive generation
$$t_i \sim \text{sample}(\text{logits}_i)$$

### Temperature
$$\text{logits}_{scaled} = \frac{\text{logits}}{T}$$

### Top-k
Keep the top k logits and set the rest to $-\infty$.

### Top-p
Keep the smallest set of tokens whose cumulative probability exceeds p.
`;

export const finetuneEquations = String.raw`
### Sequence construction
$$\text{sequence} = [\text{prompt}] + [\text{response}]$$

### Masked loss
$$m_i = 1 \text{ for response tokens, } 0 \text{ otherwise}$$

$$\mathcal{L} = \frac{\sum_i m_i \cdot \mathcal{L}_i}{\sum_i m_i}$$
`;

export const loraEquations = String.raw`
### LoRA adaptation
$$W_{effective} = W + \frac{\alpha}{r} (B A)$$

$$A \in \mathbb{R}^{r \times d_{in}}, \; B \in \mathbb{R}^{d_{out} \times r}$$
`;
