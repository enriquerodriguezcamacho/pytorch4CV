"""
Tests de F14 · Inferencia y despliegue.
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from helpers import DummyCNN, assert_close, load
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

inf = load("fundamentals/14_inference_deployment/inference.py")
dep = load("fundamentals/14_inference_deployment/deployment.py")

C, H, W = 3, 8, 8


class ConstantModel(nn.Module):
    """Predice siempre la misma clase."""

    def __init__(self, label: int, num_classes: int = 3) -> None:
        super().__init__()
        self.label = label
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.one_hot(torch.full((x.shape[0],), self.label), self.num_classes).float() * 5


# --------------------------------------------------------------------------- #
# Inferencia                                                                  #
# --------------------------------------------------------------------------- #


def test_predict_eval_and_no_grad() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(C * H * W, 4), nn.Dropout(0.9))
    model.train()
    x = torch.rand(5, C, H, W)
    labels, conf = inf.predict(model, x)
    assert not model.training, "Pasa a eval (si no, el Dropout cambia la salida)."
    assert labels.shape == (5,) and labels.dtype == torch.int64
    probs = F.softmax(model(x), dim=1)
    assert_close(conf, probs.max(1).values)
    assert torch.equal(labels, probs.argmax(1))
    assert not conf.requires_grad


def test_batched_predict_tensor_and_tuple_batches() -> None:
    model = DummyCNN(3)
    x = torch.rand(10, C, H, W)
    expected = model.eval()(x)
    with_labels = DataLoader(TensorDataset(x, torch.zeros(10)), batch_size=4)
    assert_close(inf.batched_predict(model, with_labels), expected, atol=1e-5)
    only_x = DataLoader(x, batch_size=3)
    assert_close(inf.batched_predict(model, only_x), expected, atol=1e-5)


def test_export_torchscript_roundtrip(tmp_path: Path) -> None:
    model = DummyCNN(3)
    example = torch.rand(2, C, H, W)
    path = tmp_path / "model.pt"
    traced = inf.export_torchscript(model, example, path)
    assert path.exists(), "Guarda el fichero."
    loaded = torch.jit.load(str(path))
    x = torch.rand(4, C, H, W)
    assert inf.outputs_match(model, traced, x)
    assert inf.outputs_match(model, loaded, x)


def test_outputs_match_detects_differences() -> None:
    a, b = DummyCNN(3), DummyCNN(3)
    x = torch.rand(2, C, H, W)
    assert inf.outputs_match(a, a, x)
    assert not inf.outputs_match(a, b, x)


# --------------------------------------------------------------------------- #
# Shadow                                                                      #
# --------------------------------------------------------------------------- #


def test_shadow_returns_champion() -> None:
    shadow = dep.ShadowDeployment(ConstantModel(0), ConstantModel(2))
    x = torch.rand(C, H, W)
    assert all(shadow.predict(x) == 0 for _ in range(5)), "Siempre responde el campeón."
    assert shadow.log == [(0, 2)] * 5
    assert shadow.agreement() == 0.0


def test_shadow_agreement() -> None:
    assert dep.ShadowDeployment(ConstantModel(1), ConstantModel(1)).agreement() == 0.0
    shadow = dep.ShadowDeployment(ConstantModel(1), ConstantModel(1))
    shadow.predict(torch.rand(C, H, W))
    assert shadow.agreement() == 1.0


# --------------------------------------------------------------------------- #
# A/B                                                                         #
# --------------------------------------------------------------------------- #


def test_ab_route_is_seeded() -> None:
    a = dep.ABTest(ConstantModel(0), ConstantModel(1), split=0.5, seed=3)
    b = dep.ABTest(ConstantModel(0), ConstantModel(1), split=0.5, seed=3)
    assert [a.route() for _ in range(20)] == [b.route() for _ in range(20)]
    gen = torch.Generator().manual_seed(3)
    expected = ["candidate" if torch.rand(1, generator=gen).item() < 0.5 else "champion" for _ in range(20)]
    c = dep.ABTest(ConstantModel(0), ConstantModel(1), split=0.5, seed=3)
    assert [c.route() for _ in range(20)] == expected


def test_ab_split_extremes_and_counts() -> None:
    x = torch.rand(C, H, W)
    always = dep.ABTest(ConstantModel(0), ConstantModel(1), split=1.0)
    assert always.predict(x, label=1) == ("candidate", 1)
    never = dep.ABTest(ConstantModel(0), ConstantModel(1), split=0.0)
    assert never.predict(x, label=1) == ("champion", 0)
    ab = dep.ABTest(ConstantModel(0), ConstantModel(1), split=0.3, seed=1)
    for _ in range(400):
        ab.predict(x, label=1)
    assert sum(ab.requests.values()) == 400
    assert 80 < ab.requests["candidate"] < 160
    assert ab.hits["champion"] == 0 and ab.hits["candidate"] == ab.requests["candidate"]
    hits_before = dict(ab.hits)
    for _ in range(10):
        ab.predict(x)
    assert ab.hits == hits_before, "Sin etiqueta no cambian los aciertos."
    assert sum(ab.requests.values()) == 410


def test_two_proportion_z_test() -> None:
    z, p = dep.two_proportion_z_test(50, 100, 50, 100)
    assert z == 0.0 and p == pytest.approx(1.0)
    z, p = dep.two_proportion_z_test(40, 100, 60, 100)
    pooled = 0.5
    se = math.sqrt(pooled * pooled * 2 / 100)
    assert z == pytest.approx(0.2 / se)
    assert p == pytest.approx(math.erfc(z / math.sqrt(2)))
    assert p < 0.05, "Diferencia significativa."
    z_neg, p_neg = dep.two_proportion_z_test(60, 100, 40, 100)
    assert z_neg == pytest.approx(-z) and p_neg == pytest.approx(p)
    assert dep.two_proportion_z_test(10, 10, 5, 5) == (0.0, 1.0)


# --------------------------------------------------------------------------- #
# Canary                                                                      #
# --------------------------------------------------------------------------- #


def test_canary_promotes_good_candidate() -> None:
    canary = dep.CanaryDeployment(
        ConstantModel(0), ConstantModel(1), fractions=(0.2, 0.5, 1.0), step_requests=5, min_accuracy=0.8
    )
    x = torch.rand(C, H, W)
    assert canary.fraction == 0.2
    for _ in range(300):
        canary.predict(x, label=1)
    assert canary.fraction == 1.0 and not canary.rolled_back
    assert canary.predict(x, label=1) == ("candidate", 1), "Al final atiende todo el tráfico."


def test_canary_rolls_back_bad_candidate() -> None:
    canary = dep.CanaryDeployment(
        ConstantModel(0), ConstantModel(2), fractions=(0.5, 1.0), step_requests=3, min_accuracy=0.5
    )
    x = torch.rand(C, H, W)
    served = [canary.predict(x, label=0)[0] for _ in range(100)]
    assert canary.rolled_back and canary.fraction == 0.0
    assert served.count("candidate") == 3, "Rollback tras la primera etapa."
    assert all(name == "champion" for name in served[-20:])


def test_canary_stage_counters() -> None:
    canary = dep.CanaryDeployment(
        ConstantModel(0), ConstantModel(1), fractions=(1.0,), step_requests=4, min_accuracy=0.5
    )
    x = torch.rand(C, H, W)
    for _ in range(3):
        canary.predict(x, label=1)
    assert canary.stage_requests == 3 and canary.stage_hits == 3
    canary.predict(x, label=1)
    assert canary.stage_requests == 0, "Se reinicia al acabar la etapa."
    assert canary.stage == 0, "No pasa de la última etapa."
