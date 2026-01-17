import torch
import torch.nn.functional as F
import os
from typing import List, Optional, Callable
from tqdm import tqdm
import sys

try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    from lm_eval import evaluator, tasks
except ImportError as e:
    import traceback
    traceback.print_exc()
    raise ImportError(f"Error importing lm-eval: {e}. Please install lm-eval>=0.4.0")

from pretraining.model.model_loader import load_model_from_checkpoint
from pretraining.tokenization.tokenizer import BPETokenizer, SentencePieceTokenizer, SimpleBPETokenizer, CharacterTokenizer

class CustomEvalLM(LM):
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        max_length: int = 2048,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.batch_size_per_gpu = batch_size
        self._max_length = max_length

        print(f"Loading checkpoint from {checkpoint_path}...")
        # Note: load_model_from_checkpoint returns (model, cfg, checkpoint_dict)
        self.model, self.cfg, checkpoint = load_model_from_checkpoint(checkpoint_path, self.device)
        self.model.eval()

        # Determine tokenizer type from checkpoint if possible
        tokenizer_type = checkpoint.get("tokenizer_type", "gpt2") # Default to gpt2 if unknown
        print(f"Tokenizer type from checkpoint: {tokenizer_type}")

        # Load tokenizer
        # If tokenizer_path is not provided, try to find it in the checkpoint directory
        if tokenizer_path is None:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            # Try common tokenizer names
            for name in ["tokenizer.model", "tokenizer.json", "vocab.json"]:
                path = os.path.join(checkpoint_dir, name)
                if os.path.exists(path):
                    tokenizer_path = path
                    break
        
        self.tokenizer = None
        if tokenizer_path and os.path.exists(tokenizer_path):
            print(f"Loading tokenizer from {tokenizer_path}...")
            if tokenizer_path.endswith(".model"):
                self.tokenizer = SentencePieceTokenizer(model_path=tokenizer_path)
            elif tokenizer_path.endswith(".json"):
                 self.tokenizer = SimpleBPETokenizer(model_path=tokenizer_path)
        
        # Fallback to tiktoken/BPE if no file found but type is standard
        if self.tokenizer is None:
            if tokenizer_type in ["gpt2", "tiktoken", "bpe-tiktoken"]:
                print(f"Using standard BPETokenizer (gpt2)")
                self.tokenizer = BPETokenizer(model_name="gpt2")
            elif tokenizer_type == "character":
                 print("Warning: Character tokenizer detected but no vocab file found. This might fail.")
            else:
                 print(f"Warning: Could not load tokenizer for type {tokenizer_type}")

        if self.tokenizer is None:
             raise ValueError("Failed to load tokenizer. Please check checkpoint or provide tokenizer_path.")

        self.tokenizer_type = "custom"
        
        # Determine max length from config or use default
        model_n_ctx = getattr(self.cfg, "n_ctx", max_length)
        self._max_length = model_n_ctx
        print(f"Model context length (n_ctx): {self._max_length}")

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device_name(self):
        return str(self.device)

    def tok_encode(self, string: str):
        if hasattr(self.tokenizer, "encode"):
            return self.tokenizer.encode(string)
        else:
             raise NotImplementedError("Tokenizer encode method not found")

    def tok_decode(self, tokens):
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(tokens)
        else:
            raise NotImplementedError("Tokenizer decode method not found")

    def loglikelihood(self, requests):
        res = []
        import json
        
        # Calculate chunks first to know total
        chunks = [requests[i : i + self.batch_size] for i in range(0, len(requests), self.batch_size)]
        total_chunks = len(chunks)
        
        # We attach a callback via a static variable or pass it differently?
        # Since this is subclassing LM, we can't easily change signature.
        # But we can set a property on the instance before calling evaluator.simple_evaluate
        progress_callback = getattr(self, "progress_callback", None)

        for i, chunk in enumerate(tqdm(chunks, desc="Evaluating")):
            
            if progress_callback:
                progress_callback(i, total_chunks)

            inputs = []
            targets = []
            
            for req in chunk:
                if hasattr(req, "args"):
                    context, continuation = req.args
                else:
                    context, continuation = req

                # Encode context and continuation
                context_enc = self.tok_encode(context)
                continuation_enc = self.tok_encode(continuation)
                
                # Full sequence: context + continuation
                full_enc = context_enc + continuation_enc
                
                # Input: full sequence
                # Target: full sequence, but we only care about the continuation part for loss
                inputs.append(torch.tensor(full_enc))
                targets.append(torch.tensor(full_enc))

            # Pad and batch
            # We need to pad to the maximum length in this batch
            max_len = max([len(x) for x in inputs])
            
            # Ensure we don't exceed model context length
            if max_len > self.max_length:
                print(f"Warning: Truncating input from {max_len} to {self.max_length}")
                max_len = self.max_length
                
            # Use 0 as pad token? Or something else?
            # self.tokenizer.pad_token_id if available, else 0
            pad_token = 0 
            
            padded_inputs = torch.full((len(inputs), max_len), pad_token, dtype=torch.long)
            
            for k, x in enumerate(inputs):
                # Truncate request if it's too long
                if len(x) > max_len:
                     padded_inputs[k, :] = x[:max_len]
                else:
                     padded_inputs[k, :len(x)] = x.clone()

            # Move to device
            padded_inputs = padded_inputs.to(self.device)
            
            with torch.no_grad():
                # Forward pass
                # logits: [batch, max_len, vocab_size]
                logits = self.model(padded_inputs)
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                # Compute log probs
                # We need log_softmax
                log_probs = F.log_softmax(logits, dim=-1)

            for k, req in enumerate(chunk):
                if hasattr(req, "args"):
                    context, continuation = req.args
                else:
                    context, continuation = req
                
                # Re-calculate lengths
                context_len = len(self.tok_encode(context))
                continuation_len = len(self.tok_encode(continuation))
                full_len = context_len + continuation_len
                
                if continuation_len == 0:
                    res.append((0.0, True))
                    continue
                    
                # logits slice: [context_len-1 : full_len-1]
                # target slice: [context_len : full_len]
                
                start_idx = max(0, context_len - 1)
                end_idx = full_len - 1
                
                # Extract relevant logits and targets
                # logits are aligned such that logits[t] predicts tokens[t+1]
                # so logits[context_len-1] predicts tokens[context_len] (first token of continuation)
                
                relevant_logits = log_probs[k, start_idx:end_idx]
                relevant_targets = padded_inputs[k, context_len:full_len] # These are already on device
                
                # Gather log probs of target tokens
                # relevant_logits: [seq_len, vocab_size]
                # relevant_targets: [seq_len]
                
                # gather expects index to have same dim as input except at dim
                relevant_targets = relevant_targets.unsqueeze(-1) # [seq_len, 1]
                
                token_log_probs = torch.gather(relevant_logits, -1, relevant_targets).squeeze(-1)
                
                # Sum log probs for the continuation
                total_log_prob = token_log_probs.sum().item()
                
                # Is greedy? (Evaluating if the most likely token matches target - simple greedy check)
                # Not strictly required for loglikelihood but lm-eval asks for (loglikelihood, is_greedy)
                greedy_tokens = relevant_logits.argmax(dim=-1)
                is_greedy = (greedy_tokens == relevant_targets.squeeze(-1)).all().item()
                
                res.append((total_log_prob, is_greedy))
                
        return res
    
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("loglikelihood_rolling not implemented")
        
    def generate_until(self, requests):
        raise NotImplementedError("generate_until not implemented")


class CustomEvaluator:
    def __init__(self, checkpoint_path: str, device: str = "cuda", batch_size: int = 8):
        self.lm = CustomEvalLM(
            checkpoint_path=checkpoint_path,
            device=device,
            batch_size=batch_size
        )

    def evaluate(self, tasks_list: List[str], limit: Optional[float] = None, progress_callback: Optional[Callable] = None):
        """
        Run evaluation on specified tasks.
        
        Args:
            tasks_list: List of task names
            limit: Limit number of examples per task (optional)
            progress_callback: Function(current, total) -> None
        """
        
        # Attach callback to LM instance so loglikelihood can use it
        self.lm.progress_callback = progress_callback
        
        print(f"Running evaluation on tasks: {tasks_list}")
        
        # Run evaluation
        results = evaluator.simple_evaluate(
            model=self.lm,
            tasks=tasks_list,
            limit=limit,
        )
        
        # Normalize keys (strip aggregation suffix like ,none)
        raw_results = results["results"]
        normalized_results = {}
        
        for task, metrics in raw_results.items():
            normalized_results[task] = {}
            for key, value in metrics.items():
                # lm-eval returns keys like "acc,none", "acc_stderr,none"
                # We want just "acc"
                if "," in key:
                    new_key = key.split(",")[0]
                    normalized_results[task][new_key] = value
                else:
                    normalized_results[task][key] = value
                    
        return normalized_results
