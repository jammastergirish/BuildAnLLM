"""Dataset for Supervised Fine-Tuning (SFT) with prompt/response pairs."""

import torch
import pandas as pd
from typing import Tuple
from pathlib import Path


class SFTDataset:
    """
    Dataset for supervised fine-tuning (SFT) with prompt/response pairs.

    **Key Concepts:**

    1. **Supervised Fine-Tuning (SFT)**: After pre-training on raw text, we fine-tune
       the model on structured prompt/response pairs. This teaches the model to
       follow instructions and generate appropriate responses.

    2. **Loss Masking**: Unlike pre-training where we compute loss on all tokens,
       in SFT we only compute loss on response tokens (mask == 1). This prevents
       the model from learning to repeat the prompt and focuses learning on generating
       good responses.

    3. **Sequence Format**: Each training example is formatted as:
       [prompt_tokens] + [response_tokens]
       The model learns to predict the response given the prompt.

    **Shape Flow:**
    - Input CSV: prompt (str), response (str)
    - After tokenization: prompt_tokens (List[int]), response_tokens (List[int])
    - After concatenation: full_tokens (List[int])
    - After padding/truncation: full_tokens (List[int], length = max_length)
    - After shifting: X (List[int], length = max_length-1), Y (List[int], length = max_length-1)
    - Final tensors: X [num_sequences, max_length-1], Y [num_sequences, max_length-1],
                     masks [num_sequences, max_length-1]
    """
    
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 512,
        train_split: float = 0.9,
        mask_prompt: bool = True,
    ):
        """
        Args:
            csv_path: Path to CSV file with 'prompt' and 'response' columns
            tokenizer: Tokenizer instance (must match pre-trained model tokenizer)
            max_length: Maximum sequence length
            train_split: Fraction of data for training
            mask_prompt: If True, only compute loss on response tokens
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_split = train_split
        self.mask_prompt = mask_prompt
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Handle both column name formats: prompt/response or instruction/output
        # This allows users to use either format, including instruction tuning formats
        if 'prompt' in df.columns and 'response' in df.columns:
            self.prompts = df['prompt'].tolist()
            self.responses = df['response'].tolist()
        elif 'instruction' in df.columns and 'output' in df.columns:
            # Support instruction/output format (common in instruction tuning datasets)
            self.prompts = df['instruction'].tolist()
            self.responses = df['output'].tolist()
        else:
            raise ValueError(
                "CSV must have either ('prompt', 'response') or ('instruction', 'output') columns. "
                "You can format your data however you like - including with instruction templates already applied."
            )
        
        print(f"Loaded {len(self.prompts)} prompt/response pairs")
        
        # Validate that we have data
        if len(self.prompts) == 0:
            raise ValueError("CSV file is empty! Please provide at least one prompt/response pair.")
        
        # Validate that responses are not all empty
        # Handle NaN values from pandas (empty strings may be read as NaN)
        empty_response_mask = [
            pd.isna(r) or (isinstance(r, str) and not r.strip())
            for r in self.responses
        ]
        empty_responses = sum(1 for is_empty in empty_response_mask if is_empty)
        if empty_responses == len(self.responses):
            raise ValueError(
                "All responses are empty! Please ensure your CSV has non-empty response values. "
                "The model needs response tokens to learn from."
            )
        if empty_responses > 0:
            print(f"Warning: {empty_responses} out of {len(self.responses)} responses are empty. "
                  "These will be skipped during training.")
            filtered_pairs = [
                (prompt, response)
                for prompt, response, is_empty in zip(
                    self.prompts, self.responses, empty_response_mask
                )
                if not is_empty
            ]
            if filtered_pairs:
                self.prompts, self.responses = map(list, zip(*filtered_pairs))
        
        # Create sequences
        self.X, self.Y, self.masks = self._create_sequences()
        
        # Validate that we have response tokens after processing
        total_response_tokens = self.masks.sum().item()
        if total_response_tokens == 0:
            raise ValueError(
                "No response tokens found after processing! This could happen if:\n"
                "1. All prompts are too long (exceeding max_length)\n"
                "2. All responses are empty\n"
                "3. max_length is too small\n"
                "Please check your data and increase max_length if needed."
            )
        
        # Split into train/val
        self.X_train, self.Y_train, self.masks_train, \
        self.X_val, self.Y_val, self.masks_val = self._split_data()
    
    def _create_sequences(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create (input, target, mask) pairs from prompt/response.

        Returns:
            X: [num_sequences, max_length-1] - input token sequences
            Y: [num_sequences, max_length-1] - target token sequences (shifted by 1)
            masks: [num_sequences, max_length-1] - loss mask (1 for response, 0 for prompt)
        """
        X = []
        Y = []
        # Mask: 1 for response tokens (compute loss), 0 for prompt tokens (ignore)
        # This allows us to only learn from the response, not the prompt
        masks = []

        for prompt, response in zip(self.prompts, self.responses):
            # Format: [prompt_tokens] + [response_tokens]
            # Tokenize
            # prompt_tokens: List[int] - token IDs for the prompt
            # response_tokens: List[int] - token IDs for the response
            prompt_tokens = self.tokenizer.encode(str(prompt))
            response_tokens = self.tokenizer.encode(str(response))

            # Combine: prompt + response
            # full_tokens: List[int] - concatenated prompt and response tokens
            full_tokens = prompt_tokens + response_tokens

            # Truncate if too long
            if len(full_tokens) > self.max_length:
                # Keep prompt, truncate response (we want to learn from responses)
                max_response_len = self.max_length - len(prompt_tokens) - 1
                if max_response_len > 0:
                    full_tokens = prompt_tokens + response_tokens[:max_response_len]
                    prompt_len = len(prompt_tokens)
                else:
                    # Prompt too long, truncate it (rare case)
                    prompt_len = self.max_length - 1
                    full_tokens = prompt_tokens[:prompt_len] + response_tokens[:1]
            else:
                prompt_len = len(prompt_tokens)

            # Pad or truncate to max_length
            # mask: List[int] - 0 for prompt positions (if masking), 1 for response positions
            if len(full_tokens) < self.max_length:
                # Pad with 0 (adjust if your tokenizer has a special padding token)
                padding = [0] * (self.max_length - len(full_tokens))
                full_tokens = full_tokens + padding
                if self.mask_prompt:
                    # Mask: 0s for prompt, 1s for response, 0s for padding
                    mask = ([0] * prompt_len +
                            [1] * (len(full_tokens) - prompt_len - len(padding)) +
                            [0] * len(padding))
                else:
                    # Mask: 1s for prompt + response, 0s for padding
                    mask = ([1] * (len(full_tokens) - len(padding)) +
                            [0] * len(padding))
            else:
                full_tokens = full_tokens[:self.max_length]
                if self.mask_prompt:
                    # Mask: 0s for prompt, 1s for response
                    mask = [0] * prompt_len + [1] * (len(full_tokens) - prompt_len)
                else:
                    # Mask: 1s for prompt + response
                    mask = [1] * len(full_tokens)
                mask = mask[:self.max_length]

            # Create input and target (shifted by 1, same as pre-training)
            # This is autoregressive: at position i, predict token at position i+1
            # input_seq: List[int] - tokens [0, 1, ..., max_length-2]
            # target_seq: List[int] - tokens [1, 2, ..., max_length-1]
            # mask_seq: List[int] - mask [1:] (shifted to match target positions)
            input_seq = full_tokens[:-1]
            target_seq = full_tokens[1:]
            mask_seq = mask[1:]  # Shift mask to match target positions

            X.append(input_seq)
            Y.append(target_seq)
            masks.append(mask_seq)

        # X: [num_sequences, max_length-1] - input sequences
        # Y: [num_sequences, max_length-1] - target sequences
        # masks: [num_sequences, max_length-1] - loss masks
        return (torch.tensor(X, dtype=torch.long),
                torch.tensor(Y, dtype=torch.long),
                torch.tensor(masks, dtype=torch.float))
    
    def _split_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                    torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split data into train and validation sets.

        Returns:
            X_train: [num_train, max_length-1] - training inputs
            Y_train: [num_train, max_length-1] - training targets
            masks_train: [num_train, max_length-1] - training masks
            X_val: [num_val, max_length-1] - validation inputs
            Y_val: [num_val, max_length-1] - validation targets
            masks_val: [num_val, max_length-1] - validation masks
        """
        # self.X: [num_sequences, max_length-1]
        split_idx = int(self.train_split * len(self.X))
        # X_train: [num_train, max_length-1]
        # Y_train: [num_train, max_length-1]
        # masks_train: [num_train, max_length-1]
        X_train = self.X[:split_idx]
        Y_train = self.Y[:split_idx]
        masks_train = self.masks[:split_idx]
        # X_val: [num_val, max_length-1]
        # Y_val: [num_val, max_length-1]
        # masks_val: [num_val, max_length-1]
        X_val = self.X[split_idx:]
        Y_val = self.Y[split_idx:]
        masks_val = self.masks[split_idx:]
        return X_train, Y_train, masks_train, X_val, Y_val, masks_val
    
    def get_train_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get training data with masks."""
        return self.X_train, self.Y_train, self.masks_train
    
    def get_val_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get validation data with masks."""
        return self.X_val, self.Y_val, self.masks_val
    
    def print_info(self):
        """Print dataset information."""
        print(f"Total pairs: {len(self.prompts)}")
        print(f"Max length: {self.max_length}")
        print(f"Train: {len(self.X_train)} sequences, Val: {len(self.X_val)} sequences")
        print(f"Mask prompt: {self.mask_prompt}")
