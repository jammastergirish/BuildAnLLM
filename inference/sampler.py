"""Text generation sampler with support for temperature, top-k, and top-p sampling.

This module implements autoregressive text generation from transformer models with
support for various sampling strategies and KV caching for efficient inference.

Sampling Strategies:
- Temperature: Scales logits before softmax (higher = more random)
- Top-k: Only sample from top k most likely tokens
- Top-p (Nucleus): Sample from tokens until cumulative probability > p
- Combined: Can use temperature + top-k + top-p together

Design Decision: KV Caching
- Cache K/V tensors from previous tokens to avoid recomputation
- Only compute Q, K, V for new tokens each step
- Dramatically faster for long sequences (10-100x speedup)
"""

from typing import Optional
import torch
import torch.nn.functional as F


class TransformerSampler:
    """Sampler for generating text from a trained transformer model.

    Supports multiple sampling strategies and efficient KV caching for fast generation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: torch.device,
    ):
        """Initialize sampler.

        Args:
            model: Trained transformer model
            tokenizer: Tokenizer for encoding/decoding text
            device: Device to run generation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def _apply_top_k(
        self,
        logits: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """Apply top-k filtering to logits.

        Only keeps the top k most likely tokens, sets all others to -inf.
        This prevents sampling very unlikely tokens.

        Args:
            logits: Logits tensor [vocab_size]
            top_k: Number of top tokens to keep

        Returns:
            Filtered logits [vocab_size] (unlikely tokens set to -inf)
        """
        vocab_size = logits.size(-1)
        if top_k <= 0:
            return logits
        top_k = min(top_k, vocab_size)
        # Get threshold: k-th highest logit value
        # topk_values: [top_k] - top k logit values
        # topk_values[..., -1] is the k-th highest (lowest of top k)
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        # Set all tokens below threshold to -inf
        indices_to_remove = logits < threshold
        logits = logits.clone()  # Avoid modifying original
        logits[indices_to_remove] = float("-inf")
        return logits

    def _apply_top_p(
        self,
        logits: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits.

        Keeps tokens until cumulative probability exceeds top_p.
        This is adaptive: considers more tokens when distribution is flat.

        Formula:
        1. Sort logits descending
        2. Compute cumulative probabilities
        3. Keep tokens until cumulative prob > top_p
        4. Set rest to -inf

        Args:
            logits: Logits tensor [vocab_size]
            top_p: Cumulative probability threshold (e.g., 0.9)

        Returns:
            Filtered logits [vocab_size] (low-probability tokens set to -inf)
        """
        # Sort logits descending
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        # Compute cumulative probabilities
        # cumulative_probs: [vocab_size] - cumulative sum of sorted probabilities
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Find tokens to remove (cumulative prob > top_p)
        # sorted_indices_to_remove: [vocab_size] - boolean mask
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask: keep first token that exceeds threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Map back to original indices
        # indices_to_remove: [vocab_size] - boolean mask in original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            0, sorted_indices, sorted_indices_to_remove
        )

        # Set removed tokens to -inf
        logits = logits.clone()  # Avoid modifying original
        logits[indices_to_remove] = float("-inf")
        return logits

    def _split_model_output(
        self,
        result,
    ) -> tuple[torch.Tensor, Optional[list]]:
        """Extract logits and optional cache from model output."""
        if not isinstance(result, tuple):
            return result, None
        logits = result[0]
        cache = None
        for item in result[1:]:
            if isinstance(item, (list, tuple)):
                cache = item
                break
        return logits, cache

    def _process_prompt(
        self,
        tokens_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[list], bool, int]:
        """Process initial prompt and initialize KV cache.

        Args:
            tokens_tensor: Prompt tokens [1, prompt_len]

        Returns:
            Tuple of (logits, kv_cache, use_cache, start_pos) where:
            - logits: [vocab_size] - logits for last position (not temperature-scaled)
            - kv_cache: KV cache list or None
            - use_cache: Whether cache is available
            - start_pos: Starting position for next token
        """
        prompt_len = tokens_tensor.shape[1]

        # Try to get model predictions with cache support
        try:
            result = self.model(tokens_tensor, cache=None, start_pos=0)
            logits, kv_cache = self._split_model_output(result)
            use_cache = kv_cache is not None
        except TypeError:
            # Backward compatibility: model doesn't support cache
            logits = self.model(tokens_tensor)
            kv_cache = None
            use_cache = False

        # Extract logits for last position (temperature applied later)
        logits = logits[0, -1, :]
        start_pos = prompt_len

        return logits, kv_cache, use_cache, start_pos

    def _generate_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float]
    ) -> torch.Tensor:
        """Generate next token from logits with sampling strategies.

        Applies temperature, top-k, and top-p filtering, then samples.

        Args:
            logits: Logits tensor [vocab_size]
            temperature: Sampling temperature (higher = more random)
            top_k: Optional top-k filtering
            top_p: Optional top-p (nucleus) filtering

        Returns:
            Next token ID [1]
        """
        if temperature <= 0:
            if top_k is not None:
                logits = self._apply_top_k(logits, top_k)
            if top_p is not None:
                logits = self._apply_top_p(logits, top_p)
            return torch.argmax(logits, dim=-1, keepdim=True)

        # Apply temperature scaling
        # Formula: scaled_logits = logits / temperature
        # temperature < 1: more focused (deterministic)
        # temperature = 1: balanced
        # temperature > 1: more creative (random)
        logits = logits / temperature

        # Apply top-k filtering (if specified)
        if top_k is not None:
            logits = self._apply_top_k(logits, top_k)

        # Apply top-p (nucleus) filtering (if specified)
        if top_p is not None:
            logits = self._apply_top_p(logits, top_p)

        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token

    def _update_cache(
        self,
        tokens_tensor: torch.Tensor,
        kv_cache: Optional[list],
        start_pos: int,
        use_cache: bool
    ) -> tuple[torch.Tensor, Optional[list], bool]:
        """Update KV cache and get logits for current sequence.

        Args:
            tokens_tensor: Current token sequence [1, seq_len]
            kv_cache: Current KV cache or None
            start_pos: Starting position for RoPE
            use_cache: Whether to use cache

        Returns:
            Tuple of (logits, updated_kv_cache, use_cache) where:
            - logits: [vocab_size] - logits for last token (not temperature-scaled)
            - updated_kv_cache: Updated cache or None
            - use_cache: Whether cache is still available
        """
        if use_cache and kv_cache is not None:
            # Process only the new token with cache (efficient!)
            # tokens_tensor[-1:] gets the last token: [1, 1]
            new_token_tensor = tokens_tensor[:, -1:]  # [1, 1]
            try:
                result = self.model(
                    new_token_tensor, cache=kv_cache, start_pos=start_pos)
                logits, kv_cache = self._split_model_output(result)
                if kv_cache is not None:
                    return logits[0, -1, :], kv_cache, True
                return logits[0, -1, :], None, False
            except TypeError:
                # Backward compatibility: model doesn't support cache
                logits = self.model(new_token_tensor)
                return logits[0, -1, :], None, False
        # Fallback: process full sequence (no cache)
        result = self.model(tokens_tensor)
        logits, kv_cache = self._split_model_output(result)
        if kv_cache is not None:
            return logits[0, -1, :], kv_cache, True
        return logits[0, -1, :], None, False

    @torch.no_grad()
    def sample(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Starting text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens
            top_p: If set, use nucleus sampling (top_p cumulative probability)

        Returns:
            Generated text (prompt + generated tokens)
        """
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)

        # Get context length from model config
        n_ctx = self.model.cfg.n_ctx if hasattr(self.model, 'cfg') else 1024

        # Truncate prompt if it exceeds context length
        if len(tokens) > n_ctx:
            tokens = tokens[-n_ctx:]  # Keep only the last n_ctx tokens

        # Convert to tensor with batch dimension
        tokens_tensor = torch.tensor(
            [tokens], dtype=torch.long, device=self.device)

        # Process prompt and initialize cache
        logits, kv_cache, use_cache, start_pos = self._process_prompt(
            tokens_tensor)

        # Generate new tokens
        for _ in range(max_new_tokens):
            # Apply temperature and sampling strategies to get next token
            next_token = self._generate_next_token(
                logits, temperature, top_k, top_p)

            # Append to sequence
            tokens_tensor = torch.cat(
                [tokens_tensor, next_token.unsqueeze(0)], dim=1)

            # Check if we need to truncate (shouldn't happen often with cache)
            if tokens_tensor.shape[1] > n_ctx:
                tokens_tensor = tokens_tensor[:, -n_ctx:]
                kv_cache = None
                use_cache = False
                start_pos = 0

            # Update cache and get logits for next iteration
            logits, kv_cache, use_cache = self._update_cache(
                tokens_tensor, kv_cache, start_pos, use_cache
            )

            # Update start_pos for next iteration
            start_pos += 1

        # Decode and return
        generated_tokens = tokens_tensor[0].tolist()
        # Clamp tokens to tokenizer's vocabulary size to avoid KeyError
        if hasattr(self.tokenizer, 'vocab_size'):
            vocab_size = self.tokenizer.vocab_size
            generated_tokens = [min(t, vocab_size - 1) for t in generated_tokens]
        return self.tokenizer.decode(generated_tokens)

    @torch.no_grad()
    def sample_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> list[str]:
        """Generate text for multiple prompts in a batch.

        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_k: Optional top-k filtering
            top_p: Optional top-p filtering

        Returns:
            List of generated text strings
        """
        results = []
        for prompt in prompts:
            results.append(
                self.sample(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            )
        return results
