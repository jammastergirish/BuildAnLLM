# Architecture Extensions Guide

## Current Implementation Status

Your codebase currently supports:
- **GPT**: Learned positional embeddings, LayerNorm, GELU activation
- **LLaMA**: RoPE (Rotary Position Embedding), RMSNorm, SwiGLU activation
- **OLMo**: ALiBi (Attention with Linear Biases), LayerNorm, SwiGLU activation

## Architectural Version Differences

### ⚠️ **Current Issue: No Version Differentiation**

Your current implementation treats all LLaMA versions the same, but there ARE important differences:

#### LLaMA 1 vs LLaMA 2 vs LLaMA 3:
1. **RoPE Theta (rope_theta)**:
   - LLaMA 1: `10000.0` (what you currently use)
   - LLaMA 2: `10000.0` (same)
   - LLaMA 3: `500000.0` (5x larger - improves long context handling)
   
2. **Grouped Query Attention (GQA)**:
   - LLaMA 1: Standard multi-head attention
   - LLaMA 2: GQA in larger models (7B+)
   - LLaMA 3: GQA standard across sizes
   
3. **Vocabulary Size**:
   - LLaMA 1: 32,000 tokens
   - LLaMA 2: 32,000 tokens  
   - LLaMA 3: 128,256 tokens (much larger)

4. **Context Window**:
   - LLaMA 1: 2,048 tokens
   - LLaMA 2: 4,096 tokens
   - LLaMA 3: 128K tokens (with some variants)

#### LLaMA 4 (Future):
- **Mixture-of-Experts (MoE)**: Major architectural change - would require significant refactoring
- **Multimodal**: Text + vision - completely different input pipeline
- **10M token context**: Would need different attention mechanisms

### Recommendations for Version Support

**Easy to add:**
1. Add `llama_version` field to `ModelConfig`
2. Set `rope_theta` based on version:
   - LLaMA 1/2: `10000.0`
   - LLaMA 3: `500000.0`
3. Add vocabulary size configuration
4. Add GQA support (moderate effort)

**Harder to add:**
- LLaMA 4 MoE architecture (would require major refactoring)
- Multimodal capabilities

## Other Models to Consider Adding

### 🟢 **Easy to Add** (Similar Architecture)

#### 1. **Mistral 7B** ⭐ Recommended
- **Architecture**: Very similar to LLaMA 2
- **Key differences**: 
  - Sliding Window Attention (SWA) - attention only over last N tokens
  - Otherwise identical: RoPE, RMSNorm, SwiGLU
- **Effort**: Low-Medium (need to implement SWA)
- **Why add**: Popular, well-documented, minimal changes needed

#### 2. **Qwen (Alibaba)**
- **Architecture**: LLaMA-like
- **Key differences**: 
  - RoPE with different scaling
  - RMSNorm (same as LLaMA)
- **Effort**: Low (mostly config changes)

#### 3. **Phi (Microsoft)**
- **Architecture**: Similar to GPT but with some tweaks
- **Key differences**:
  - Uses learned positional embeddings (like GPT)
  - Different normalization
- **Effort**: Low-Medium

#### 4. **Gemma (Google)**
- **Architecture**: Based on Gemini technology
- **Key differences**:
  - RoPE (like LLaMA)
  - GeGLU activation (variant of SwiGLU)
  - RMSNorm
- **Effort**: Low-Medium (GeGLU is similar to SwiGLU)

### 🟡 **Medium Difficulty** (Some Architectural Changes)

#### 5. **GPT-J / GPT-NeoX**
- **Architecture**: GPT-like but with some differences
- **Key differences**:
  - Parallel attention/MLP (not sequential)
  - Rotary embeddings (RoPE) instead of learned
  - Different initialization
- **Effort**: Medium (need to change block structure)

#### 6. **PaLM-style**
- **Architecture**: Similar to GPT but with:
  - SwiGLU activation
  - Parallel layers
  - Different normalization
- **Effort**: Medium

### 🔴 **Hard to Add** (Major Architectural Changes)

#### 7. **Mixture-of-Experts (MoE) Models**
- **Examples**: Mixtral, DBRX, LLaMA 4
- **Key differences**:
  - Multiple expert networks
  - Router to select experts
  - Sparse activation
- **Effort**: High (major refactoring needed)

#### 8. **Multimodal Models**
- **Examples**: LLaVA, GPT-4V, LLaMA 4
- **Key differences**:
  - Vision encoder
  - Cross-modal attention
  - Different input pipeline
- **Effort**: Very High (completely different architecture)

## Implementation Priority Recommendations

### Phase 1: Version Differentiation (High Priority)
1. Add LLaMA version support (1, 2, 3)
2. Implement configurable `rope_theta`
3. Add vocabulary size configuration
4. **Effort**: Low, **Impact**: High (correctness)

### Phase 2: Easy Models (Medium Priority)
1. **Mistral 7B** - Add Sliding Window Attention
2. **Gemma** - Add GeGLU activation variant
3. **Qwen** - Mostly config changes
4. **Effort**: Low-Medium, **Impact**: Medium

### Phase 3: Architectural Improvements (Lower Priority)
1. Grouped Query Attention (GQA) for LLaMA 2/3
2. Parallel attention/MLP (GPT-J style)
3. **Effort**: Medium, **Impact**: Medium

### Phase 4: Advanced (Future)
1. MoE support
2. Multimodal capabilities
3. **Effort**: Very High, **Impact**: High (but complex)

## Quick Implementation Guide

### Adding a New Model Architecture

1. **Add to `Architecture` enum** in `config.py`:
```python
class Architecture(str, Enum):
    GPT = "gpt"
    LLAMA = "llama"
    OLMO = "olmo"
    MISTRAL = "mistral"  # New
```

2. **Add config method** in `ModelConfig`:
```python
@classmethod
def mistral_small(cls):
    return cls(
        architecture=Architecture.MISTRAL,
        # ... config values
    )
```

3. **Update model.py** to handle new architecture:
```python
if cfg.architecture == Architecture.MISTRAL:
    # Mistral-specific setup (e.g., SWA)
```

4. **Update components** as needed:
   - `attention.py` - for attention mechanism changes
   - `mlp.py` - for activation function changes
   - `transformer_block.py` - for block structure changes

5. **Update train.py** to support new architecture in CLI

## Specific Model Details

### Mistral 7B
- **RoPE**: Same as LLaMA (theta=10000)
- **Normalization**: RMSNorm
- **Activation**: SwiGLU
- **Special**: Sliding Window Attention (4,096 window)
- **Implementation**: Modify attention to mask beyond window

### Gemma
- **RoPE**: Similar to LLaMA
- **Normalization**: RMSNorm  
- **Activation**: GeGLU (gate uses GELU instead of SiLU)
- **Implementation**: New MLP class with GeGLU

### Qwen
- **RoPE**: Dynamic scaling
- **Normalization**: RMSNorm
- **Activation**: SwiGLU
- **Implementation**: Mostly config, some RoPE tweaks

## Testing Strategy

When adding new models:
1. Start with small config (like your `_small()` variants)
2. Test forward pass shapes match expected
3. Compare outputs with reference implementation if available
4. Test training loop works
5. Verify loss decreases during training

## Resources

- **Hugging Face Model Cards**: Great source for architectural details
- **Papers**: Original papers have exact specifications
- **Reference Implementations**: Check official repos for details


