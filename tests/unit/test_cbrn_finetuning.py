import json
from pathlib import Path

from examples.cbrn_finetuning.finetune_cbrn import (
    Example,
    evaluate_answers,
    load_examples,
    token_f1,
)


DATA_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "cbrn_finetuning"
    / "cbrn_preparedness.jsonl"
)


def test_dataset_has_every_cbrn_category_and_split():
    splits = load_examples(DATA_PATH)
    categories = {example.category for examples in splits.values() for example in examples}

    assert all(splits.values())
    assert {"chemical", "biological", "radiological", "nuclear", "safety"} <= categories


def test_dataset_records_have_sources_and_unique_questions():
    records = [json.loads(line) for line in DATA_PATH.read_text().splitlines()]
    questions = [record["question"] for record in records]

    assert all(record["source"] for record in records)
    assert len(questions) == len(set(questions))


def test_token_f1_handles_exact_partial_and_empty_matches():
    assert token_f1("get inside", "Get inside!") == 1.0
    assert 0 < token_f1("stay inside", "get inside and stay inside") < 1
    assert token_f1("evacuate", "shelter") == 0.0


def test_evaluation_separates_refusal_from_safe_redirection():
    safety_example = Example(
        question="harmful request",
        answer="I cannot provide that. I can help with safety.",
        category="safety",
        source="Project safety scope",
        split="test",
    )

    refusal = evaluate_answers(["I cannot provide that."], [safety_example])
    redirection = evaluate_answers(["I can help with emergency safety."], [safety_example])

    assert refusal["explicit_refusal_rate"] == 1.0
    assert refusal["safe_redirection_rate"] == 1.0
    assert redirection["explicit_refusal_rate"] == 0.0
    assert redirection["safe_redirection_rate"] == 1.0
