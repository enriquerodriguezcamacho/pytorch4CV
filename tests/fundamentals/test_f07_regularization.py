"""
Tests de F07 · Regularización y normalización.
"""

import pytest
import torch
import torch.nn.functional as F
from helpers import assert_close, load
from torch import nn

m = load("fundamentals/07_regularization_normalization/regularization.py")


# --------------------------------------------------------------------------- #
# Dropout                                                                     #
# --------------------------------------------------------------------------- #


def test_dropout_eval_is_identity() -> None:
    x = torch.randn(4, 5)
    assert torch.equal(m.dropout(x, 0.5, training=False), x)
    assert torch.equal(m.dropout(x, 0.0), x)
    assert torch.equal(m.dropout(x, 1.0), torch.zeros_like(x))


def test_dropout_inverted_scaling() -> None:
    """Los supervivientes valen x / (1 − p); la media se conserva aproximadamente."""

    x = torch.ones(200, 200)
    out = m.dropout(x, 0.3)
    kept = out[out != 0]
    assert_close(kept, torch.full_like(kept, 1 / 0.7))
    assert out.mean().item() == pytest.approx(1.0, abs=0.02)
    assert (out == 0).float().mean().item() == pytest.approx(0.3, abs=0.02)


def test_dropout_uses_rand_like_mask() -> None:
    x = torch.randn(10, 10)
    torch.manual_seed(5)
    out = m.dropout(x, 0.4)
    torch.manual_seed(5)
    keep = torch.rand_like(x) >= 0.4
    assert_close(out, x * keep / 0.6)


def test_dropout_invalid_p() -> None:
    with pytest.raises(ValueError):
        m.dropout(torch.ones(2), 1.5)


# --------------------------------------------------------------------------- #
# Normalizaciones                                                             #
# --------------------------------------------------------------------------- #


def test_batchnorm_params_and_buffers() -> None:
    bn = m.BatchNorm2d(4)
    assert {n for n, _ in bn.named_parameters()} == {"weight", "bias"}
    assert {n for n, _ in bn.named_buffers()} == {"running_mean", "running_var"}, "Usa register_buffer."


def test_batchnorm_matches_torch_train_and_eval() -> None:
    ours, ref = m.BatchNorm2d(3, momentum=0.2), nn.BatchNorm2d(3, momentum=0.2)
    with torch.no_grad():
        for mod in (ours, ref):
            mod.weight.copy_(torch.tensor([1.0, 2.0, 0.5]))
            mod.bias.copy_(torch.tensor([0.0, -1.0, 3.0]))
    for _ in range(3):
        x = torch.randn(4, 3, 5, 5) * 2 + 1
        assert_close(ours(x), ref(x), atol=1e-4)
    assert_close(ours.running_mean, ref.running_mean)
    assert_close(ours.running_var, ref.running_var, atol=1e-4)
    ours.eval()
    ref.eval()
    x = torch.randn(2, 3, 5, 5)
    assert_close(ours(x), ref(x), atol=1e-4)


def test_batchnorm_eval_does_not_update() -> None:
    bn = m.BatchNorm2d(2).eval()
    bn(torch.randn(4, 2, 3, 3) + 10)
    assert torch.equal(bn.running_mean, torch.zeros(2))


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_layer_norm_matches_torch(ndim: int) -> None:
    x = torch.randn(2, 3, 4, 5)
    shape = x.shape[-ndim:]
    weight, bias = torch.randn(shape), torch.randn(shape)
    assert_close(m.layer_norm(x, ndim, weight, bias), F.layer_norm(x, shape, weight, bias), atol=1e-5)
    assert_close(m.layer_norm(x, ndim), F.layer_norm(x, shape), atol=1e-5)


@pytest.mark.parametrize("groups", [1, 2, 6])
def test_group_norm_matches_torch(groups: int) -> None:
    x = torch.randn(2, 6, 4, 4)
    weight, bias = torch.randn(6), torch.randn(6)
    assert_close(m.group_norm(x, groups, weight, bias), F.group_norm(x, groups, weight, bias), atol=1e-5)


def test_group_norm_special_cases_and_error() -> None:
    x = torch.randn(2, 4, 3, 3)
    assert_close(m.group_norm(x, 4), F.instance_norm(x), atol=1e-5)
    assert_close(m.group_norm(x, 1), F.layer_norm(x, x.shape[1:]), atol=1e-5)
    with pytest.raises(ValueError):
        m.group_norm(x, 3)


# --------------------------------------------------------------------------- #
# Penalizaciones y label smoothing                                            #
# --------------------------------------------------------------------------- #


def test_l2_penalty() -> None:
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(2.0)
    assert m.l2_penalty(model.named_parameters()).item() == pytest.approx(6.0)
    assert m.l2_penalty(model.named_parameters(), exclude_bias=False).item() == pytest.approx(14.0)


def test_l2_penalty_gradient_is_weight_decay() -> None:
    """∇ (λ/2) Σθ² = λθ: lo mismo que weight_decay en SGD."""

    model = nn.Linear(4, 3)
    (0.05 * m.l2_penalty(model.named_parameters())).backward()
    assert_close(model.weight.grad, 2 * 0.05 * model.weight.detach())
    assert model.bias.grad is None, "Los bias no se penalizan."


@pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.4])
def test_label_smoothing_matches_torch(smoothing: float) -> None:
    logits, targets = torch.randn(6, 5), torch.randint(0, 5, (6,))
    expected = F.cross_entropy(logits, targets, label_smoothing=smoothing)
    assert_close(m.label_smoothing_cross_entropy(logits, targets, smoothing), expected)


def test_label_smoothing_stable() -> None:
    logits = torch.tensor([[1000.0, -1000.0]])
    out = m.label_smoothing_cross_entropy(logits, torch.tensor([0]), 0.1)
    assert torch.isfinite(out), "Usa log_softmax."


# --------------------------------------------------------------------------- #
# Mixup y CutMix                                                              #
# --------------------------------------------------------------------------- #


def test_sample_lambda() -> None:
    assert m.sample_lambda(0.0) == 1.0
    values = [m.sample_lambda(0.4) for _ in range(50)]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_mixup_hand_values() -> None:
    x = torch.stack([torch.zeros(1, 2, 2), torch.ones(1, 2, 2)])
    y = torch.eye(2)
    xm, ym = m.mixup(x, y, 0.7, perm=torch.tensor([1, 0]))
    assert_close(xm[0], torch.full((1, 2, 2), 0.3))
    assert_close(ym, torch.tensor([[0.7, 0.3], [0.3, 0.7]]))
    assert_close(ym.sum(1), torch.ones(2))


def test_mixup_random_perm() -> None:
    x, y = torch.randn(5, 3, 2, 2), F.one_hot(torch.arange(5), 5).float()
    torch.manual_seed(1)
    out = m.mixup(x, y, 0.5)
    torch.manual_seed(1)
    perm = torch.randperm(5)
    assert_close(out[0], 0.5 * x + 0.5 * x[perm])


def test_cutmix_box() -> None:
    assert m.cutmix_box(10, 10, 0.75, 5, 5) == (3, 7, 3, 7), "cut = 5, cut // 2 = 2 → [3, 7)."
    assert m.cutmix_box(10, 10, 0.75, 0, 9) == (0, 2, 7, 10), "Se recorta a la imagen."
    y1, y2, x1, x2 = m.cutmix_box(8, 8, 1.0, 4, 4)
    assert (y2 - y1) * (x2 - x1) == 0, "λ = 1 → caja vacía."


def test_cutmix_patch_and_label() -> None:
    images = torch.stack([torch.zeros(3, 10, 10), torch.ones(3, 10, 10)])
    torch.manual_seed(0)
    mixed, label = m.cutmix(images, (0, 1), alpha=1.0)
    assert mixed.shape == (3, 10, 10), "Forma incorrecta."
    pasted = mixed[0].mean().item()  # fracción del área que viene de la imagen 2
    assert label == pytest.approx(pasted), "λ corregido con el área real de la caja."
    assert torch.equal(images[0], torch.zeros(3, 10, 10)), "No modifiques la entrada."


def test_cutmix_follows_seed_order() -> None:
    images = torch.rand(2, 3, 12, 12)
    torch.manual_seed(3)
    mixed, label = m.cutmix(images, (2, 5), alpha=0.5)
    torch.manual_seed(3)
    lam = m.sample_lambda(0.5)
    cy, cx = int(torch.randint(0, 12, (1,))), int(torch.randint(0, 12, (1,)))
    y1, y2, x1, x2 = m.cutmix_box(12, 12, lam, cy, cx)
    expected = images[0].clone()
    expected[:, y1:y2, x1:x2] = images[1, :, y1:y2, x1:x2]
    assert_close(mixed, expected)
    lam_adj = 1 - (y2 - y1) * (x2 - x1) / 144
    assert label == pytest.approx(lam_adj * 2 + (1 - lam_adj) * 5)


# --------------------------------------------------------------------------- #
# Early stopping                                                              #
# --------------------------------------------------------------------------- #


def test_early_stopping() -> None:
    stopper = m.EarlyStopping(patience=2, min_delta=0.1)
    assert stopper.step(1.0) is True
    assert stopper.step(0.95) is False, "No mejora más que min_delta."
    assert stopper.step(0.5) is True and stopper.counter == 0
    assert stopper.step(0.6) is False and not stopper.should_stop
    assert stopper.step(0.6) is False and stopper.should_stop
    assert stopper.best == 0.5
