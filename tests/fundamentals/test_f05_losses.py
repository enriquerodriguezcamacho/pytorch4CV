"""
Tests de F05 · Funciones de pérdida.
"""

import math

import pytest
import torch
import torch.nn.functional as F
import torchvision
from helpers import assert_close, load

m = load("fundamentals/05_losses/losses.py")

NUM_CLASSES = 4
BATCH = 6
IGNORE = 255
REDUCTIONS = ["mean", "sum", "none"]


# ----------------------------------------------------------------- reduce_loss
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_reduce_loss(reduction: str) -> None:
    """Las tres reducciones coinciden con mean/sum/identidad."""

    x = torch.tensor([1.0, 2.0, 3.0, 6.0])
    expected = {"mean": torch.tensor(3.0), "sum": torch.tensor(12.0), "none": x}
    assert_close(m.reduce_loss(x, reduction), expected[reduction])


def test_reduce_loss_invalid() -> None:
    """Una reducción desconocida lanza ValueError."""

    with pytest.raises(ValueError):
        m.reduce_loss(torch.ones(3), "media")


# --------------------------------------------------------------- cross_entropy
def test_cross_entropy_hand_value() -> None:
    """Logits iguales → CE = log(C)."""

    logits = torch.zeros(2, NUM_CLASSES)
    target = torch.tensor([0, 3])
    out = m.cross_entropy(logits, target)
    assert out.shape == (), "La CE con reduction='mean' debe ser escalar."
    assert torch.isclose(out, torch.tensor(math.log(NUM_CLASSES))), "Valor incorrecto."


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("use_weight", [False, True])
def test_cross_entropy_vs_torch(
    reduction: str, label_smoothing: float, use_weight: bool
) -> None:
    """Coincide con F.cross_entropy (pesos, label smoothing y reducciones)."""

    logits = torch.randn(BATCH, NUM_CLASSES)
    target = torch.randint(0, NUM_CLASSES, (BATCH,))
    weight = torch.rand(NUM_CLASSES) + 0.5 if use_weight else None
    out = m.cross_entropy(
        logits, target, weight=weight, label_smoothing=label_smoothing, reduction=reduction
    )
    ref = F.cross_entropy(
        logits, target, weight=weight, label_smoothing=label_smoothing, reduction=reduction
    )
    assert out.shape == ref.shape, "Forma incorrecta."
    assert_close(out, ref)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.2])
@pytest.mark.parametrize("use_weight", [False, True])
def test_cross_entropy_segmentation_ignore_index(
    label_smoothing: float, use_weight: bool
) -> None:
    """Entrada (B,C,H,W), target (B,H,W) con píxeles 255 ignorados."""

    logits = torch.randn(2, NUM_CLASSES, 5, 6)
    target = torch.randint(0, NUM_CLASSES, (2, 5, 6))
    target[0, :2] = IGNORE
    weight = torch.rand(NUM_CLASSES) + 0.5 if use_weight else None
    for reduction in REDUCTIONS:
        out = m.cross_entropy(
            logits,
            target,
            weight=weight,
            ignore_index=IGNORE,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
        ref = F.cross_entropy(
            logits,
            target,
            weight=weight,
            ignore_index=IGNORE,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
        assert_close(out, ref)
    none = m.cross_entropy(logits, target, ignore_index=IGNORE, reduction="none")
    assert none.shape == (2, 5, 6), "Con 'none' la forma debe ser (B, H, W)."
    assert (none[0, :2] == 0).all(), "Los píxeles ignorados deben valer 0."


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_cross_entropy_soft_targets(label_smoothing: float) -> None:
    """Target de probabilidades (MixUp/CutMix) igual que en PyTorch."""

    logits = torch.randn(BATCH, NUM_CLASSES)
    target = F.softmax(torch.randn(BATCH, NUM_CLASSES), dim=1)
    weight = torch.rand(NUM_CLASSES) + 0.5
    out = m.cross_entropy(logits, target, weight=weight, label_smoothing=label_smoothing)
    ref = F.cross_entropy(logits, target, weight=weight, label_smoothing=label_smoothing)
    assert_close(out, ref)


def test_cross_entropy_grad() -> None:
    """El gradiente respecto a los logits es softmax - one_hot (dividido por B)."""

    logits = torch.randn(3, NUM_CLASSES, requires_grad=True)
    target = torch.tensor([1, 0, 2])
    m.cross_entropy(logits, target).backward()
    expected = (F.softmax(logits, 1) - F.one_hot(target, NUM_CLASSES)) / 3
    assert_close(logits.grad, expected.detach())


# ------------------------------------------------------------------------ BCE
def test_bce_hand_value() -> None:
    """logit 0 → σ = 0.5 → BCE = log 2 para cualquier etiqueta."""

    out = m.binary_cross_entropy_with_logits(torch.zeros(4), torch.tensor([0.0, 1, 0, 1]))
    assert torch.isclose(out, torch.tensor(math.log(2))), "Valor incorrecto."


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("use_pos_weight", [False, True])
def test_bce_vs_torch(reduction: str, use_pos_weight: bool) -> None:
    """Coincide con F.binary_cross_entropy_with_logits (multi-etiqueta)."""

    logits = torch.randn(BATCH, NUM_CLASSES) * 3
    target = torch.randint(0, 2, (BATCH, NUM_CLASSES)).float()
    pos_weight = torch.tensor([1.0, 2.0, 5.0, 0.5]) if use_pos_weight else None
    out = m.binary_cross_entropy_with_logits(logits, target, pos_weight, reduction)
    ref = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction=reduction
    )
    assert_close(out, ref)


def test_bce_stable_large_logits() -> None:
    """Sin inf/nan con logits enormes (por eso se usa la versión con logits)."""

    logits = torch.tensor([1000.0, -1000.0, 1000.0, -1000.0])
    target = torch.tensor([1.0, 0.0, 0.0, 1.0])
    out = m.binary_cross_entropy_with_logits(logits, target, reduction="none")
    assert torch.isfinite(out).all(), "La BCE debe ser estable numéricamente."
    assert_close(out, torch.tensor([0.0, 0.0, 1000.0, 1000.0]))


# ------------------------------------------------------- smooth L1 y Huber
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_smooth_l1_vs_torch(beta: float, reduction: str) -> None:
    """Coincide con F.smooth_l1_loss para varios β."""

    pred, target = torch.randn(5, 4) * 2, torch.randn(5, 4)
    out = m.smooth_l1_loss(pred, target, beta=beta, reduction=reduction)
    assert_close(out, F.smooth_l1_loss(pred, target, beta=beta, reduction=reduction))


def test_smooth_l1_hand_value() -> None:
    """|d| = 0.5 → 0.125 ; |d| = 3 → 2.5 (β = 1)."""

    out = m.smooth_l1_loss(torch.tensor([0.5, 3.0]), torch.zeros(2), reduction="none")
    assert_close(out, torch.tensor([0.125, 2.5]))


@pytest.mark.parametrize("delta", [0.5, 1.0, 3.0])
def test_huber_vs_torch(delta: float) -> None:
    """Huber coincide con F.huber_loss y con δ · smooth_l1(β = δ)."""

    pred, target = torch.randn(5, 4) * 3, torch.randn(5, 4)
    out = m.huber_loss(pred, target, delta=delta)
    assert_close(out, F.huber_loss(pred, target, delta=delta))
    assert_close(out, delta * F.smooth_l1_loss(pred, target, beta=delta))


# --------------------------------------------------------------- distillation
@pytest.mark.parametrize("temperature", [1.0, 2.0, 4.0])
def test_distillation_kl_vs_torch(temperature: float) -> None:
    """T² · KL(batchmean) igual que con F.kl_div."""

    s, t = torch.randn(BATCH, NUM_CLASSES), torch.randn(BATCH, NUM_CLASSES)
    out = m.distillation_kl(s, t, temperature)
    ref = (
        F.kl_div(
            F.log_softmax(s / temperature, 1),
            F.softmax(t / temperature, 1),
            reduction="batchmean",
        )
        * temperature**2
    )
    assert out.shape == (), "Debe ser escalar."
    assert_close(out, ref)


def test_distillation_kl_zero_and_no_teacher_grad() -> None:
    """KL = 0 si estudiante = profesor y el profesor no recibe gradiente."""

    z = torch.randn(3, NUM_CLASSES)
    assert torch.isclose(m.distillation_kl(z, z.clone()), torch.tensor(0.0), atol=1e-6)
    s = torch.randn(3, NUM_CLASSES, requires_grad=True)
    t = torch.randn(3, NUM_CLASSES, requires_grad=True)
    m.distillation_kl(s, t).backward()
    assert s.grad is not None and s.grad.abs().sum() > 0, "El estudiante debe aprender."
    assert t.grad is None, "El profesor debe estar desacoplado (detach)."


# ---------------------------------------------------------------------- focal
@pytest.mark.parametrize("alpha", [0.25, -1.0])
@pytest.mark.parametrize("gamma", [0.0, 2.0])
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_focal_vs_torchvision(alpha: float, gamma: float, reduction: str) -> None:
    """Coincide con torchvision.ops.sigmoid_focal_loss."""

    logits = torch.randn(BATCH, NUM_CLASSES) * 2
    target = torch.randint(0, 2, (BATCH, NUM_CLASSES)).float()
    out = m.sigmoid_focal_loss(logits, target, alpha, gamma, reduction)
    ref = torchvision.ops.sigmoid_focal_loss(logits, target, alpha, gamma, reduction)
    assert_close(out, ref)


def test_focal_downweights_easy_examples() -> None:
    """Un ejemplo fácil (p_t ≈ 0.95) pesa mucho menos que con BCE."""

    logits, target = torch.tensor([3.0]), torch.tensor([1.0])
    fl = m.sigmoid_focal_loss(logits, target, alpha=-1, gamma=2.0)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    assert fl < 0.01 * bce, "La focal loss debe atenuar los ejemplos fáciles."


# ----------------------------------------------------------------------- dice
def test_dice_perfect_and_hand_value() -> None:
    """Predicción casi perfecta → ~0; caso binario calculado a mano."""

    target = torch.tensor([[[0, 1], [1, 0]]])
    logits = F.one_hot(target, 2).permute(0, 3, 1, 2).float() * 50
    out = m.dice_loss(logits, target, smooth=0.0)
    assert out.shape == (), "Debe ser escalar."
    assert out < 1e-4, "Predicción perfecta → Dice loss ≈ 0."
    # logits nulos → p = 0.5 en todo: Dice_c = 2·1 / (2 + 2) = 0.5 → pérdida 0.5
    out = m.dice_loss(torch.zeros(1, 2, 2, 2), target, smooth=0.0)
    assert torch.isclose(out, torch.tensor(0.5)), "Valor incorrecto."


def test_dice_ignore_index() -> None:
    """Los píxeles ignorados no cambian el Dice."""

    logits = torch.randn(2, 3, 4, 4)
    target = torch.randint(0, 3, (2, 4, 4))
    target_ign = target.clone()
    target_ign[:, 0] = IGNORE
    logits_mod = logits.clone()
    logits_mod[:, :, 0] = torch.randn(2, 3, 4) * 10
    a = m.dice_loss(logits, target_ign, ignore_index=IGNORE)
    b = m.dice_loss(logits_mod, target_ign, ignore_index=IGNORE)
    assert_close(a, b)
    assert 0 <= a <= 1, "La Dice loss está en [0, 1]."


# ------------------------------------------------------------------------ GIoU
def test_giou_vs_torchvision() -> None:
    """Coincide con torchvision.ops.generalized_box_iou_loss."""

    xy = torch.rand(8, 2) * 10
    b1 = torch.cat([xy, xy + torch.rand(8, 2) * 5 + 0.1], 1)
    xy = torch.rand(8, 2) * 10
    b2 = torch.cat([xy, xy + torch.rand(8, 2) * 5 + 0.1], 1)
    out = m.generalized_iou_loss(b1, b2)
    assert out.shape == (8,), "Forma incorrecta."
    ref = torchvision.ops.generalized_box_iou_loss(b1, b2, reduction="none")
    assert_close(out, ref, atol=1e-4)


def test_giou_hand_values() -> None:
    """Misma caja → 0 ; cajas disjuntas lejanas → cerca de 2."""

    b = torch.tensor([[0.0, 0, 2, 2], [0, 0, 1, 1]])
    c = torch.tensor([[0.0, 0, 2, 2], [9, 9, 10, 10]])
    out = m.generalized_iou_loss(b, c)
    assert torch.isclose(out[0], torch.tensor(0.0), atol=1e-5), "Cajas iguales → 0."
    # C = 10×10 = 100, unión = 2 → GIoU = 0 - 98/100 → pérdida 1.98
    assert torch.isclose(out[1], torch.tensor(1.98), atol=1e-5), "Valor incorrecto."


# --------------------------------------------------------------------- triplet
@pytest.mark.parametrize("p", [1.0, 2.0])
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_triplet_vs_torch(p: float, reduction: str) -> None:
    """Coincide con F.triplet_margin_loss."""

    a, pos, neg = torch.randn(BATCH, 8), torch.randn(BATCH, 8), torch.randn(BATCH, 8)
    out = m.triplet_margin_loss(a, pos, neg, margin=0.7, p=p, reduction=reduction)
    ref = F.triplet_margin_loss(a, pos, neg, margin=0.7, p=p, reduction=reduction)
    assert_close(out, ref)


# -------------------------------------------------------------------- InfoNCE
def test_info_nce_hand_value() -> None:
    """B = 2 con vistas ortogonales idénticas: CE de logits [1/τ, 0]."""

    z = torch.eye(2)
    tau = 0.5
    out = m.info_nce(z, z, temperature=tau)
    expected = -math.log(math.exp(1 / tau) / (math.exp(1 / tau) + 1))
    assert torch.isclose(out, torch.tensor(expected)), "Valor incorrecto."


def test_info_nce_vs_reference_and_symmetric() -> None:
    """Igual a la CE sobre similitudes coseno; la simétrica promedia ambas."""

    z1, z2 = torch.randn(BATCH, 16), torch.randn(BATCH, 16) * 3
    sim = F.cosine_similarity(z1[:, None], z2[None], dim=-1) / 0.1
    labels = torch.arange(BATCH)
    assert_close(m.info_nce(z1, z2, 0.1), F.cross_entropy(sim, labels))
    sym = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))
    assert_close(m.info_nce(z1, z2, 0.1, symmetric=True), sym)
    # invariante a la escala de los embeddings
    assert_close(m.info_nce(z1 * 5, z2, 0.1), m.info_nce(z1, z2, 0.1))


# ---------------------------------------------------------- DistillationLoss
ALPHA = 0.3
TEMPERATURE = 3.0


def test_distillation_loss_combination() -> None:
    """α·CE + (1-α)·T²·KL."""

    s, t = torch.randn(BATCH, NUM_CLASSES), torch.randn(BATCH, NUM_CLASSES)
    y = torch.randint(0, NUM_CLASSES, (BATCH,))
    loss_fn = m.DistillationLoss(alpha=ALPHA, temperature=TEMPERATURE)
    out = loss_fn(s, t, y)
    kl = F.kl_div(
        F.log_softmax(s / TEMPERATURE, 1), F.softmax(t / TEMPERATURE, 1), reduction="batchmean"
    )
    expected = ALPHA * F.cross_entropy(s, y) + (1 - ALPHA) * TEMPERATURE**2 * kl
    assert_close(out, expected)
    assert_close(loss_fn._hard_loss(s, y), F.cross_entropy(s, y))
    assert_close(loss_fn._soft_loss(s, t), TEMPERATURE**2 * kl)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_distillation_loss_extremes(alpha: float) -> None:
    """α = 1 → solo CE ; α = 0 → solo la parte suave."""

    s, t = torch.randn(BATCH, NUM_CLASSES), torch.randn(BATCH, NUM_CLASSES)
    y = torch.randint(0, NUM_CLASSES, (BATCH,))
    loss_fn = m.DistillationLoss(alpha=alpha, temperature=TEMPERATURE)
    expected = loss_fn._hard_loss(s, y) if alpha == 1.0 else loss_fn._soft_loss(s, t)
    assert_close(loss_fn(s, t, y), expected)


def test_distillation_loss_gradients() -> None:
    """Solo el estudiante recibe gradiente."""

    s = torch.randn(BATCH, NUM_CLASSES, requires_grad=True)
    t = torch.randn(BATCH, NUM_CLASSES, requires_grad=True)
    y = torch.randint(0, NUM_CLASSES, (BATCH,))
    m.DistillationLoss(ALPHA, TEMPERATURE)(s, t, y).backward()
    assert s.grad is not None and torch.norm(s.grad) > 0, "Falta gradiente."
    assert t.grad is None, "El profesor no debe recibir gradiente."


@pytest.mark.parametrize("alpha, temperature", [(-0.1, 2.0), (1.5, 2.0), (0.5, 0.0)])
def test_distillation_loss_invalid(alpha: float, temperature: float) -> None:
    """Parámetros fuera de rango → ValueError."""

    with pytest.raises(ValueError):
        m.DistillationLoss(alpha, temperature)


# --------------------------------------------------- CombinedSegmentationLoss
CE_WEIGHT = 0.6
DICE_WEIGHT = 1.4


def _seg_batch() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn(2, NUM_CLASSES, 6, 6)
    target = torch.randint(0, NUM_CLASSES, (2, 6, 6))
    target[1, -1] = IGNORE
    return logits, target


def test_combined_segmentation_components() -> None:
    """λ_dice = 0 → CE oficial ; λ_ce = 0 → Dice."""

    logits, target = _seg_batch()
    ce_only = m.CombinedSegmentationLoss(1.0, 0.0, ignore_index=IGNORE)
    assert_close(ce_only(logits, target), F.cross_entropy(logits, target, ignore_index=IGNORE))
    dice_only = m.CombinedSegmentationLoss(0.0, 1.0, ignore_index=IGNORE)
    assert_close(dice_only(logits, target), m.dice_loss(logits, target, ignore_index=IGNORE))


@pytest.mark.parametrize("which", ["ce", "dice"])
def test_combined_segmentation_linear_scaling(which: str) -> None:
    """Dividir un peso por 2 divide por 2 su contribución."""

    logits, target = _seg_batch()
    full = m.CombinedSegmentationLoss(CE_WEIGHT, DICE_WEIGHT, ignore_index=IGNORE)
    if which == "ce":
        half = m.CombinedSegmentationLoss(CE_WEIGHT / 2, DICE_WEIGHT, ignore_index=IGNORE)
        part = m.CombinedSegmentationLoss(CE_WEIGHT, 0.0, ignore_index=IGNORE)
    else:
        half = m.CombinedSegmentationLoss(CE_WEIGHT, DICE_WEIGHT / 2, ignore_index=IGNORE)
        part = m.CombinedSegmentationLoss(0.0, DICE_WEIGHT, ignore_index=IGNORE)
    diff = full(logits, target) - half(logits, target)
    assert_close(diff, part(logits, target) / 2)


def test_combined_segmentation_class_weights_and_grad() -> None:
    """Los pesos de clase se pasan a la CE y hay gradiente hacia los logits."""

    logits, target = _seg_batch()
    logits.requires_grad_(True)
    w = torch.tensor([1.0, 2.0, 0.5, 3.0])
    loss_fn = m.CombinedSegmentationLoss(1.0, 0.0, ignore_index=IGNORE, class_weights=w)
    out = loss_fn(logits, target)
    assert_close(out, F.cross_entropy(logits, target, weight=w, ignore_index=IGNORE))
    assert "class_weights" in dict(loss_fn.named_buffers()), "Registra los pesos como buffer."
    out.backward()
    assert logits.grad is not None and torch.norm(logits.grad) > 0, "Falta gradiente."
