"""scripts/diag/train_predictor.py smoke test on synthetic dumps (CPU, tiny, few epochs)."""

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "diag" / "train_predictor.py"

EXPERTS = 12
HIDDEN = 8
TOP_K = 3
TOKENS = 48


@pytest.fixture(scope="module")
def train_predictor():
    spec = importlib.util.spec_from_file_location("diag_train_predictor", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dumps(tmp_path):
    """Two layers, two chunks each, with a LEARNABLE next-layer target.

    Expert ids are a deterministic function of the sign of x[:, 0], so a trained head must
    beat chance -- which is what says the training loop actually optimizes something.
    """
    generator = torch.Generator().manual_seed(7)
    for layer_id in (0, 1):
        for chunk in (0, 1):
            x = torch.randn(TOKENS, HIDDEN, generator=generator)
            hot = (x[:, 0] > 0).long()
            base = torch.tensor([0, 1, 2]) + 3 * hot.unsqueeze(1)
            torch.save(
                {
                    "layer": layer_id,
                    "num_experts": EXPERTS,
                    "positions": list(range(chunk * TOKENS, (chunk + 1) * TOKENS)),
                    "x": x.to(torch.float16),
                    "top10_l": base.to(torch.int32),
                    "top10_l1": (base + 6 * hot.unsqueeze(1)).clamp(0, EXPERTS - 1).to(torch.int32),
                    "top10_l2": base.to(torch.int32),
                },
                tmp_path / f"x_layer{layer_id:02d}_{chunk}.pt",
            )
    return tmp_path


def test_reports_both_heads_per_layer(train_predictor, dumps, capsys):
    argv = [str(dumps), "--epochs", "25", "--batch", "16", "--targets", "l1", "--ks", "3,10"]
    assert train_predictor.main(argv) == 0
    out = capsys.readouterr().out
    lines = [line.split() for line in out.strip().splitlines()[1:]]
    assert [(row[0], row[1], row[2]) for row in lines] == [
        ("0", "l1", "linear"), ("0", "l1", "mlp"),
        ("1", "l1", "linear"), ("1", "l1", "mlp"),
    ]
    # 80/20 by position over the two concatenated chunks
    assert all(row[3] == str(int(0.8 * 2 * TOKENS)) for row in lines)
    assert all(row[4] == str(2 * TOKENS - int(0.8 * 2 * TOKENS)) for row in lines)
    # a learnable target: both heads beat the 3/12 chance rate at k=3, so the loop trains
    assert all(float(row[5]) > 0.25 for row in lines), lines
    assert all(float(row[6]) >= float(row[5]) for row in lines), "recall must grow with k"


def test_layer_filter_and_both_targets(train_predictor, dumps, capsys):
    argv = [str(dumps), "--epochs", "1", "--batch", "32", "--layers", "1"]
    assert train_predictor.main(argv) == 0
    lines = [line.split() for line in capsys.readouterr().out.strip().splitlines()[1:]]
    assert {row[0] for row in lines} == {"1"}
    assert {row[1] for row in lines} == {"l1", "l2"}


def test_missing_dumps(train_predictor, tmp_path, capsys):
    assert train_predictor.main([str(tmp_path)]) == 2
    assert "no x_layer*.pt dumps" in capsys.readouterr().out


def test_recall_at_matches_a_hand_computed_case(train_predictor):
    logits = torch.tensor([[9.0, 8.0, 0.0, 0.0], [0.0, 0.0, 9.0, 8.0]])
    ids = torch.tensor([[0, 3], [2, 3]])  # top-2 finds 1 of 2, then 2 of 2
    assert train_predictor.recall_at(logits, ids, 2) == pytest.approx(0.75)
    assert train_predictor.recall_at(logits, ids, 4) == pytest.approx(1.0)
