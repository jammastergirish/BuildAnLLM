"""Unit tests for the RL arithmetic task: format, reward and held-out split."""

import csv
import random
from collections import Counter

import pytest
from rl.arithmetic import (
    eval_problems,
    format_answer,
    format_prompt,
    is_held_out,
    parse_number,
    reward_fn,
    sample_problem,
    spell,
    write_sft_csv,
)


@pytest.mark.unit
class TestFormat:
    def test_spell_and_parse_round_trip(self):
        for n in [0, 7, 42, 100, 1998, -5]:
            assert parse_number(spell(n)) == n
        assert spell(428) == "4 2 8"
        assert parse_number("abc") is None

    def test_prompt_and_answer(self):
        assert format_prompt(42, 35) == "Q: What is 4 2 + 3 5?\nA: "
        assert format_answer(77) == "<answer>7 7</answer>"


@pytest.mark.unit
class TestReward:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("<answer>7 7</answer>", (1.5, True, True)),
            ("<answer>77</answer>", (1.5, True, True)),
            ("<answer>7 8</answer>", (0.3, True, False)),
            ("<answer>1 7 7</answer>", (0.3, True, False)),
            ("7 7", (0.2, False, False)),
            ("nonsense", (0.0, False, False)),
        ],
    )
    def test_graded_reward(self, text, expected):
        assert reward_fn(text, 77) == expected


@pytest.mark.unit
class TestHeldOut:
    def test_symmetric(self):
        for a in range(100):
            for b in range(100):
                assert is_held_out(a, b) == is_held_out(b, a)

    def test_about_one_in_ten(self):
        n = sum(is_held_out(a, b) for a in range(100) for b in range(100))
        assert 800 < n < 1200

    def test_does_not_select_by_last_digit_of_sum(self):
        # Every final digit of a + b must appear among training problems,
        # or the model never learns to write answers ending in it.
        held = [(a, b) for a in range(100) for b in range(100) if is_held_out(a, b)]
        counts = Counter((a + b) % 10 for a, b in held)
        assert len(counts) == 10
        assert max(counts.values()) < 3 * min(counts.values())

    def test_training_never_draws_held_out(self):
        rng = random.Random(0)
        for _ in range(2000):
            assert not is_held_out(*sample_problem(2, rng))

    def test_eval_sets(self):
        unseen = eval_problems(3, held_out=True)
        seen = eval_problems(3, held_out=False)
        assert len(unseen) == len(seen) == 64
        assert all(is_held_out(a, b) for a, b in unseen)
        assert not any(is_held_out(a, b) for a, b in seen)
        assert all(100 <= x <= 999 for pair in unseen + seen for x in pair)
        assert eval_problems(3, held_out=True) == unseen  # fixed, not random

    def test_sft_csv_excludes_held_out(self, tmp_path):
        path = write_sft_csv(tmp_path / "a.csv", n=500, max_digits=3)
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 500
        for row in rows:
            q = row["prompt"].split("What is ")[1].split("?")[0]
            a, b = (parse_number(x) for x in q.split(" + "))
            assert not is_held_out(a, b)
            assert reward_fn(row["response"], a + b)[2]
