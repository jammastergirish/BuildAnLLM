"""Unit tests for the GRPO trainer, on a tiny randomly initialised model."""

import pytest
import torch
from backend.app.routers.rl import tokenizer_problem
from backend.app.services.checkpoints import list_checkpoints
from config import ModelConfig
from pretraining.model.model import TransformerModel
from pretraining.model.model_loader import load_model_from_checkpoint
from pretraining.tokenization.tokenizer import BPETokenizer, CharacterTokenizer
from rl.arithmetic import format_prompt
from rl.grpo import GRPOTrainer


@pytest.fixture
def tokenizer():
    return BPETokenizer()


@pytest.fixture
def trainer(tokenizer, device):
    torch.manual_seed(0)
    cfg = ModelConfig(
        d_model=32, n_heads=2, d_head=16, d_mlp=64, n_layers=1, n_ctx=64, d_vocab=tokenizer.vocab_size
    )
    model = TransformerModel(cfg).to(device)
    return GRPOTrainer(model, tokenizer, device, "bpe-tiktoken", lr=1e-3)


@pytest.mark.unit
class TestGRPO:
    def test_cached_generation_matches_full_forward(self, trainer, tokenizer):
        """Sampling uses a KV cache; scoring uses a full forward pass. They must
        agree, or the update would score tokens from a different distribution
        than the one that sampled them."""
        ids = tokenizer.encode(format_prompt(12, 34))
        model = trainer.policy
        tokens = torch.tensor([ids] * 3, device=trainer.device)
        out = model(tokens, cache=None, start_pos=0)
        cache = out[1]
        nxt = torch.tensor([[5], [6], [7]], device=trainer.device)
        cached = model(nxt, cache=cache, start_pos=len(ids))[0][:, -1]
        full = model(torch.cat([tokens, nxt], 1))
        full = (full[0] if isinstance(full, tuple) else full)[:, -1]
        assert torch.allclose(cached, full, atol=1e-4)

    def test_round_shapes_and_advantages(self, trainer):
        out = trainer.run_round(
            train=False, digits=2, group=4, max_new=6, temperature=1.0, problems=2
        )
        assert len(out["groups"]) == 2
        assert out["update"] is None and out["step"] == 0
        for g in out["groups"]:
            assert len(g["answers"]) == 4
            advs = [a["advantage"] for a in g["answers"]]
            assert abs(sum(advs)) < 1e-3  # advantages are centred on the group mean
            for a in g["answers"]:
                assert 1 <= a["n_tokens"] <= 6
                assert a["kl"] == pytest.approx(0.0, abs=1e-5)  # policy == frozen copy

    def test_equal_scores_teach_nothing(self, trainer):
        """If every answer scores the same, every advantage is zero. At the
        start the policy equals its frozen copy, so the KL term is zero too,
        and the update leaves the weights where they were."""
        before = [p.detach().clone() for p in trainer.policy.parameters()]
        answers, _, ctx = trainer.rollout(1, 2, group=4, max_new=4, temperature=1.0)
        ctx["adv"] = torch.zeros_like(ctx["adv"])
        trainer.score_groups([ctx], [answers], train=True)
        assert all(torch.equal(b, p) for b, p in zip(before, trainer.policy.parameters()))

    def test_train_moves_weights_and_reset_restores(self, trainer):
        before = [p.detach().clone() for p in trainer.policy.parameters()]
        answers, _, ctx = trainer.rollout(1, 2, group=4, max_new=4, temperature=1.0)
        ctx["adv"] = torch.tensor([1.0, -1.0, 1.0, -1.0], device=trainer.device)
        update = trainer.score_groups([ctx], [answers], train=True)
        assert update is not None and update["grad_norm"] > 0
        assert trainer.step == 1
        assert any(not torch.equal(b, p) for b, p in zip(before, trainer.policy.parameters()))
        # the frozen copy never trains
        assert all(torch.equal(b, p) for b, p in zip(before, trainer.ref.parameters()))

        trainer.reset()
        assert trainer.step == 0
        assert all(torch.equal(b, p) for b, p in zip(before, trainer.policy.parameters()))

    def test_positive_advantage_raises_log_prob(self, trainer):
        """The core of GRPO: answers with a positive advantage become more likely."""
        answers, _, ctx = trainer.rollout(1, 2, group=4, max_new=4, temperature=1.0)
        ctx["adv"] = torch.tensor([1.5, -0.5, -0.5, -0.5], device=trainer.device)
        trainer.beta = 0.0
        lp_before = trainer.seq_logprobs(trainer.policy, ctx["ids"], ctx["n_comp"]).detach()
        for _ in range(3):
            trainer.score_groups([ctx], [answers], train=True)
        lp_after = trainer.seq_logprobs(trainer.policy, ctx["ids"], ctx["n_comp"]).detach()
        m = ctx["mask"]
        gain = ((lp_after - lp_before) * m).sum(1) / m.sum(1)
        assert gain[0] > 0
        assert gain[0] > gain[1:].max()

    def test_fixed_question_only_when_sampling(self, trainer):
        out = trainer.run_round(
            train=False, digits=2, group=2, max_new=4, temperature=1.0, problem=(11, 22)
        )
        assert out["groups"][0]["summary"]["target"] == 33

    def test_evaluate(self, trainer):
        r = trainer.evaluate(
            digits=1, samples=2, temperature=1.0, max_new=4, which="ref", held_out=True
        )
        assert r["which"] == "ref" and r["held_out"] is True
        assert 0.0 <= r["correct_rate"] <= 1.0
        assert r["n"] == 2 * len(r["problems"])

    def test_save_appears_as_rl_stage(self, trainer, tmp_path, device):
        run = tmp_path / "checkpoints" / "20260101"
        path = trainer.save_checkpoint(str(run / "rl"), {"model_type": "with_einops"})
        entries = list_checkpoints(str(tmp_path / "checkpoints"))
        assert [e["stage"] for e in entries] == ["rl"]
        model, _, ckpt = load_model_from_checkpoint(path, device)
        assert ckpt["is_rl"] and ckpt["tokenizer_type"] == "bpe-tiktoken"


@pytest.mark.unit
class TestTokenizerCheck:
    def test_tiktoken_ok(self, tokenizer):
        assert tokenizer_problem(tokenizer) is None

    def test_tokenizer_without_tags_rejected(self):
        tok = CharacterTokenizer("a plain novel with no angle brackets.")
        assert tokenizer_problem(tok) is not None
