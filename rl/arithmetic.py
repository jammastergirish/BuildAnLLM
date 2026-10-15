"""Arithmetic task for reinforcement learning: questions, prompt format and reward.

RL needs a task whose answers a program can check. Addition fits: the right
answer is an integer we can compute, so the reward is a regex and a comparison,
with no learned reward model that the policy could learn to fool.

The same prompt format is used for the SFT dataset and for RL rollouts. SFT
teaches the model the `<answer>...</answer>` tags; RL then rewards it for
putting the RIGHT number inside them.

Run `uv run python -m rl.arithmetic` to (re)write the SFT CSV.
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
from pathlib import Path

PROMPT_TEMPLATE = "Q: What is {a} + {b}?\nA: "
ANSWER_TEMPLATE = "<answer>{t}</answer>"
STOP_STRING = "</answer>"

# Numbers are written one digit at a time, with spaces: 42 becomes "4 2".
# The GPT-2 tokenizer turns a whole number like "42" into ONE token, so a small
# model can only memorise each sum as a lookup-table entry; nothing it learns
# about 42 + 35 carries over to 43 + 35. With one token per digit, the same
# column-addition skill serves every problem, so what GRPO reinforces on one
# problem helps on the others too.
ANSWER_RE = re.compile(r"<answer>\s*(-?[\d ]+?)\s*</answer>")


def spell(n: int) -> str:
    """42 -> "4 2"; -7 -> "-7"."""
    sign, digits = ("-", str(-n)) if n < 0 else ("", str(n))
    return sign + " ".join(digits)


def parse_number(text: str) -> int | None:
    """Inverse of spell(): "4 2" -> 42. None if it is not a number."""
    compact = text.replace(" ", "")
    return int(compact) if re.fullmatch(r"-?\d+", compact) else None

SFT_CSV_PATH = Path("input_data/finetuning/arithmetic.csv")

EVAL_SET_SIZE = 64


def is_held_out(a: int, b: int) -> bool:
    """True for the problems that SFT and RL never see: about 1 in 10.

    The rule is symmetric, so if 42 + 35 is held out then so is 35 + 42.
    Otherwise the model could answer a held-out problem by having memorised
    its mirror image, and the check would overstate what it learned.
    """
    lo, hi = min(a, b), max(a, b)
    # A hash, not arithmetic on a and b: a rule like (7a + 9b) % 10 would pick
    # out problems by the last digit of their sum, and then the model would
    # never see a training answer ending in that digit.
    digest = hashlib.blake2b(f"{lo},{hi}".encode(), digest_size=2).digest()
    return int.from_bytes(digest, "big") % 10 == 0


def format_prompt(a: int, b: int) -> str:
    return PROMPT_TEMPLATE.format(a=spell(a), b=spell(b))


def format_answer(target: int) -> str:
    return ANSWER_TEMPLATE.format(t=spell(target))


def sample_problem(digits: int, rng: random.Random | None = None) -> tuple[int, int]:
    """One random training problem with operands of at most `digits` digits.

    Never returns a held-out problem.
    """
    rng = rng or random
    hi = 10**digits - 1
    while True:
        a, b = rng.randint(0, hi), rng.randint(0, hi)
        if not is_held_out(a, b):
            return a, b


def eval_problems(
    digits: int, held_out: bool, n: int = EVAL_SET_SIZE
) -> list[tuple[int, int]]:
    """A fixed set of problems whose operands both have exactly `digits` digits.

    held_out=True gives problems training never sees. held_out=False gives
    problems from the training pool, for contrast: a model can get better on
    those just by memorising, so only the held-out set shows real learning.
    """
    rng = random.Random(1000 + digits)
    lo, hi = (0 if digits == 1 else 10 ** (digits - 1)), 10**digits - 1
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    # 1-digit problems are only 100 in total, so cap the attempts.
    for _ in range(100_000):
        if len(out) >= n:
            break
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if (a, b) in seen or is_held_out(a, b) != held_out:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out


def reward_fn(text: str, target: int | None) -> tuple[float, bool, bool]:
    """Verifiable reward, graded in three parts.

    - 0.2 if the right number appears anywhere (right number, wrong format)
    - 0.3 if the answer uses the <answer> tags
    - 1.0 more if the number inside the tags is correct

    The partial credit matters. GRPO learns only from differences inside a
    group. If every answer scores 0, every advantage is 0 and nothing moves.
    Partial credit separates "nearly right" from "nonsense".

    Returns (reward, format_ok, correct).
    """
    m = ANSWER_RE.search(text)
    has_num = target is not None and (
        re.search(rf"(?<!\d)(?<!\d ){re.escape(spell(target))}(?! \d)(?!\d)", text)
        or re.search(rf"(?<!\d){target}(?!\d)", text)
    )
    near = 0.2 if has_num else 0.0
    fmt = 0.3 if m else 0.0
    cor = 1.0 if (m and target is not None and parse_number(m.group(1)) == target) else 0.0
    return near + fmt + cor, fmt > 0, cor > 0


def write_sft_csv(
    path: Path = SFT_CSV_PATH,
    n: int = 20000,
    max_digits: int = 3,
    seed: int = 0,
) -> Path:
    """Write prompt/response pairs that teach the answer format.

    Mixes operand sizes from 1 digit up to `max_digits`, so the model sees easy
    cases alongside the ones RL will train on. Held-out problems are excluded.
    """
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "response"])
        for _ in range(n):
            digits = rng.randint(1, max_digits)
            a, b = sample_problem(digits, rng)
            writer.writerow([format_prompt(a, b), format_answer(a + b)])
    return path


if __name__ == "__main__":
    out = write_sft_csv()
    print(f"wrote {out}")
