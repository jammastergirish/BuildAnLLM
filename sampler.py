import torch
import torch.nn.functional as F
from typing import Optional


class TransformerSampler:
    """Sampler for generating text from a trained transformer model"""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

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
        """
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Generate tokens
        for _ in range(max_new_tokens):
            # Get model predictions
            logits = self.model(tokens_tensor)
            # Get logits for last position
            logits = logits[0, -1, :] / temperature

            # Apply top_k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Apply top_p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    0, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            tokens_tensor = torch.cat([tokens_tensor, next_token.unsqueeze(0)], dim=1)

        # Decode and return
        generated_tokens = tokens_tensor[0].tolist()
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
        """Generate text for multiple prompts in a batch"""
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

