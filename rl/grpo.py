"""Group Relative Policy Optimization (GRPO) for the models this app trains.

One round of GRPO:

1. Ask the model one question G times (a "group"), sampling at temperature.
2. Score every answer with a program (see `rl/arithmetic.py`).
3. Compare each score with the group's own mean. That difference, divided by
   the group's spread, is the answer's "advantage". No value network needed:
   the group is its own yardstick.
4. Push up the log-probability of every token in above-average answers, and
   push down every token in below-average answers.
5. Add a KL penalty that keeps the model close to a frozen copy of where it
   started, so it cannot drift into nonsense that happens to score well.

Every token in one answer gets the same advantage. GRPO never asks which token
made an answer right; across many rounds, tokens that keep appearing in good
answers get pushed up more often than down.

This takes one gradient step per batch of answers, so the policy that sampled
them is the policy being updated. The probability ratio that PPO and the GRPO
paper clip is therefore always 1, and the clip is left out.
"""

from __future__ import annotations

import copy
import os
import random
import threading
import time
from typing import Any

import torch
from rl.arithmetic import (
    STOP_STRING,
    eval_problems,
    format_prompt,
    reward_fn,
    sample_problem,
)


def _logits_of(result: Any) -> torch.Tensor:
    return result[0] if isinstance(result, tuple) else result


def _cache_of(result: Any) -> list | None:
    if not isinstance(result, tuple):
        return None
    for item in result[1:]:
        if isinstance(item, (list, tuple)):
            return item
    return None


class GRPOTrainer:
    """Holds the policy, its frozen reference copy and the optimiser."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: torch.device,
        tokenizer_type: str,
        lr: float = 1e-5,
        beta: float = 0.02,
        chunk: int = 8,
    ):
        self.policy = model
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.device = device
        self.lr = lr
        self.beta = beta
        self.chunk = chunk
        # The models here have no dropout, so eval mode changes nothing about
        # the maths; it just keeps sampling and scoring identical.
        self.policy.eval()
        # The frozen copy. The KL term measures drift from it, and the held-out
        # check uses it to measure the untrained model.
        self.ref = copy.deepcopy(model).eval()
        self.ref.requires_grad_(False)
        self.opt = self._make_optimizer()
        self.step = 0
        # One model and one optimiser, mutated in place: only one request may
        # sample or update at a time.
        self.lock = threading.Lock()

    def _make_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.policy.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=0.0)

    @property
    def n_ctx(self) -> int:
        cfg = getattr(self.policy, "cfg", None)
        return int(getattr(cfg, "n_ctx", 256))

    def reset(self) -> None:
        """Throw away everything learned: copy the frozen weights back."""
        self.policy.load_state_dict(self.ref.state_dict())
        self.opt = self._make_optimizer()
        self.step = 0

    # ------------------------------------------------------------------ sampling

    @torch.no_grad()
    def generate(
        self,
        model: torch.nn.Module,
        prompt_ids: list[int],
        group: int,
        max_new: int,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample `group` completions of one prompt in a single batch.

        Every row shares the prompt, so no padding is needed. Each row stops at
        the end of its first `</answer>`; rows that finish early keep sampling
        (the model has no end-of-sequence token) but those tokens are masked out.

        Returns completions [group, T] and a mask [group, T] of 1s up to and
        including each row's last real token.
        """
        max_new = max(1, min(max_new, self.n_ctx - len(prompt_ids)))
        tokens = torch.tensor(
            [prompt_ids] * group, dtype=torch.long, device=self.device
        )
        vocab = getattr(self.tokenizer, "vocab_size", None)

        result = model(tokens, cache=None, start_pos=0)
        logits, cache = _logits_of(result)[:, -1, :], _cache_of(result)
        generated: list[torch.Tensor] = []
        stop_at = [-1] * group  # index of each row's last real token
        for t in range(max_new):
            logits = logits.float()
            if vocab is not None and logits.shape[-1] > vocab:
                logits[:, vocab:] = float("-inf")  # ids the tokenizer cannot decode
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
            generated.append(nxt)

            done = True
            seq = torch.cat(generated, 1).tolist()
            for k in range(group):
                if stop_at[k] < 0 and STOP_STRING in self.tokenizer.decode(seq[k]):
                    stop_at[k] = t
                done = done and stop_at[k] >= 0
            if done or t == max_new - 1:
                break

            if cache is not None:
                result = model(nxt, cache=cache, start_pos=len(prompt_ids) + t)
                logits, cache = _logits_of(result)[:, -1, :], _cache_of(result)
            else:
                full = torch.cat([tokens] + generated, 1)
                logits = _logits_of(model(full))[:, -1, :]

        completions = torch.cat(generated, 1)
        n = completions.shape[1]
        ends = torch.tensor(
            [s if s >= 0 else n - 1 for s in stop_at], device=self.device
        )
        pos = torch.arange(n, device=self.device).expand(group, n)
        mask = (pos <= ends.unsqueeze(1)).float()
        return completions, mask

    def seq_logprobs(
        self, model: torch.nn.Module, ids: torch.Tensor, n_completion: int
    ) -> torch.Tensor:
        """Log-probability of each completion token: [G, n_completion]."""
        outs = []
        for i in range(0, ids.shape[0], self.chunk):
            part = ids[i : i + self.chunk]
            logits = _logits_of(model(part))[:, :-1]
            logp = torch.log_softmax(logits.float(), dim=-1)
            picked = logp.gather(2, part[:, 1:].unsqueeze(-1)).squeeze(-1)
            outs.append(picked[:, -n_completion:])
        return torch.cat(outs, 0)

    def _decode_tokens(self, ids: list[int]) -> list[str]:
        out = []
        for t in ids:
            try:
                out.append(self.tokenizer.decode([t]))
            except Exception:
                out.append(f"<{t}>")
        return out

    def rollout(
        self,
        a: int,
        b: int,
        group: int,
        max_new: int,
        temperature: float,
        model: torch.nn.Module | None = None,
    ) -> tuple[list, dict, dict]:
        """Steps 1-3: sample a group, score it, and compute advantages."""
        m = self.policy if model is None else model
        prompt = format_prompt(a, b)
        target = a + b
        prompt_ids = self.tokenizer.encode(prompt)

        t0 = time.time()
        completions, mask = self.generate(m, prompt_ids, group, max_new, temperature)
        gen_secs = time.time() - t0

        lens = mask.sum(1).long().tolist()
        texts, rewards, fmts, cors = [], [], [], []
        for k in range(group):
            ids = completions[k, : lens[k]].tolist()
            text = self.tokenizer.decode(ids)
            r, f, c = reward_fn(text, target)
            texts.append(text)
            rewards.append(r)
            fmts.append(f)
            cors.append(c)

        rew = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        # The yardstick is the group's own mean. No second network anywhere.
        mean, std = rew.mean(), rew.std()
        adv = (rew - mean) / (std + 1e-4)

        answers = []
        for k in range(group):
            ids = completions[k, : lens[k]].tolist()
            answers.append(
                {
                    "text": texts[k],
                    "tokens": self._decode_tokens(ids),
                    "n_tokens": int(lens[k]),
                    "reward": round(rewards[k], 4),
                    "format_ok": bool(fmts[k]),
                    "correct": bool(cors[k]),
                    "advantage": round(adv[k].item(), 4),
                }
            )
        prompt_t = torch.tensor(
            [prompt_ids] * group, dtype=torch.long, device=self.device
        )
        ctx = {
            "ids": torch.cat([prompt_t, completions], 1),
            "mask": mask,
            "n_comp": completions.shape[1],
            "adv": adv,
        }
        summary = {
            "question": prompt,
            "a": a,
            "b": b,
            "target": target,
            "mean": round(mean.item(), 4),
            "std": round(std.item(), 4),
            "format_rate": sum(fmts) / group,
            "correct_rate": sum(cors) / group,
            "gen_secs": round(gen_secs, 3),
        }
        return answers, summary, ctx

    # ------------------------------------------------------------------ update

    def score_groups(
        self,
        ctxs: list[dict],
        answer_lists: list[list],
        train: bool,
        beta: float | None = None,
    ) -> dict | None:
        """Steps 4-5: the loss, and one optimiser step if `train` is set.

        For each group:
            loss = mean over answers of
                   mean over that answer's tokens of
                   ( -advantage * log p(token)  +  beta * KL(policy || frozen) )
        The update averages that loss over every group in the round.

        Gradients from each group are accumulated before the single step, so
        memory holds one group's graph at a time.
        """
        b = self.beta if beta is None else beta
        if train:
            self.opt.zero_grad()
        total_loss = kl_sum = tok_sum = 0.0
        for ctx, answers in zip(ctxs, answer_lists):
            m = ctx["mask"]
            with torch.set_grad_enabled(train):
                logp = self.seq_logprobs(self.policy, ctx["ids"], ctx["n_comp"])
            with torch.no_grad():
                ref_logp = self.seq_logprobs(self.ref, ctx["ids"], ctx["n_comp"])
            # The k3 estimator of KL: always >= 0, and 0 when the two models agree.
            log_ratio = ref_logp - logp
            kl = torch.exp(log_ratio) - log_ratio - 1.0

            # adv is per ANSWER, broadcast across every token of it.
            pg = -logp * ctx["adv"].unsqueeze(1)
            per_token = pg + b * kl
            denom = m.sum(1).clamp(min=1)
            loss = ((per_token * m).sum(1) / denom).mean() / len(ctxs)
            if train:
                loss.backward()
            total_loss += loss.item()
            kl_sum += (kl.detach() * m).sum().item()
            tok_sum += m.sum().item()

            with torch.no_grad():
                mean_lp = ((logp.detach() * m).sum(1) / denom).tolist()
                mean_kl = ((kl.detach() * m).sum(1) / denom).tolist()
            for k, a in enumerate(answers):
                a["mean_logp"] = round(mean_lp[k], 4)
                a["kl"] = round(mean_kl[k], 6)

        if not train:
            return None
        params = [p for p in self.policy.parameters() if p.requires_grad]
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        self.opt.step()
        self.step += 1
        return {
            "loss": round(total_loss, 5),
            "kl": round(kl_sum / max(tok_sum, 1), 6),
            "grad_norm": round(gnorm, 4),
        }

    def run_round(
        self,
        train: bool,
        digits: int,
        group: int,
        max_new: int,
        temperature: float,
        problems: int = 1,
        beta: float | None = None,
        problem: tuple[int, int] | None = None,
    ) -> dict:
        """One round: `problems` questions, `group` answers each, one update.

        Training always draws fresh random problems. A fixed `problem` is only
        used when sampling without training.
        """
        with self.lock:
            if problem and not train:
                picks = [problem]
            else:
                picks = [sample_problem(digits) for _ in range(problems)]
            groups, ctxs = [], []
            for a, b in picks:
                answers, summary, ctx = self.rollout(a, b, group, max_new, temperature)
                groups.append({"answers": answers, "summary": summary})
                ctxs.append(ctx)
            update = self.score_groups(
                ctxs, [g["answers"] for g in groups], train, beta
            )
            n = sum(len(g["answers"]) for g in groups)
            return {
                "groups": groups,
                "update": update,
                "step": self.step,
                "trained": train,
                "format_rate": sum(x["format_ok"] for g in groups for x in g["answers"]) / n,
                "correct_rate": sum(x["correct"] for g in groups for x in g["answers"]) / n,
                "mean_reward": sum(x["reward"] for g in groups for x in g["answers"]) / n,
            }

    # ------------------------------------------------------------------ eval

    @torch.no_grad()
    def evaluate(
        self,
        digits: int,
        samples: int,
        temperature: float,
        max_new: int,
        which: str = "policy",
        held_out: bool = True,
    ) -> dict:
        """Score a fixed evaluation set: held-out problems, or training ones."""
        model = self.policy if which == "policy" else self.ref
        problems = eval_problems(digits, held_out)
        n_fmt = n_cor = n = 0
        rows = []
        with self.lock:
            for a, b in problems:
                answers, _, _ = self.rollout(
                    a, b, samples, max_new, temperature, model=model
                )
                f = sum(x["format_ok"] for x in answers)
                c = sum(x["correct"] for x in answers)
                n_fmt += f
                n_cor += c
                n += len(answers)
                rows.append(
                    {
                        "a": a,
                        "b": b,
                        "correct": c,
                        "formatted": f,
                        "n": len(answers),
                        "example": answers[0]["text"],
                    }
                )
        return {
            "which": which,
            "digits": digits,
            "held_out": held_out,
            "format_rate": n_fmt / max(n, 1),
            "correct_rate": n_cor / max(n, 1),
            "n": n,
            "problems": rows,
        }

    # ------------------------------------------------------------------ save

    def save_checkpoint(self, save_dir: str, source_checkpoint: dict) -> str:
        """Save the current policy where the inference tab can find it."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "final_model.pt")
        cfg = getattr(self.policy, "cfg", None)
        data = {
            "model_state_dict": self.policy.state_dict(),
            "iter_num": self.step,
            "cfg": cfg.to_dict() if cfg is not None else None,
            "model_type": "with_einops" if getattr(self.policy, "use_einops", True) else "without_einops",
            "tokenizer_type": self.tokenizer_type,
            "is_finetuned": True,
            "is_rl": True,
        }
        for key in ("use_lora", "lora_info"):
            if key in source_checkpoint:
                data[key] = source_checkpoint[key]
        torch.save(data, path)
        return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
