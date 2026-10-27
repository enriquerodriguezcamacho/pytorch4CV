"""
Tests de CV06 · Ataques adversarios: FGSM, PGD, I-FGSM, C&W, One Pixel Attack y defensas.
"""

import pytest
import torch
import torch.nn.functional as F
from helpers import DummyCNN, assert_close, load
from torch import nn

wb = load("cv/06_adversarial_attacks/white_box.py")
bb = load("cv/06_adversarial_attacks/black_box.py")
d = load("cv/06_adversarial_attacks/defenses.py")

BATCH, C, H, W = 4, 3, 6, 6
NUM_CLASSES = 3


def _linear_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(C * H * W, NUM_CLASSES))


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.rand(BATCH, C, H, W) * 0.6 + 0.2, torch.randint(0, NUM_CLASSES, (BATCH,))


def _manual_grad(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """∇_x CE para el modelo lineal: Wᵀ (softmax − onehot) / batch."""

    linear = model[1]
    probs = F.softmax(linear(x.flatten(1)), dim=1)
    err = (probs - F.one_hot(y, NUM_CLASSES)) / x.shape[0]
    return (err @ linear.weight).view_as(x)


# --------------------------------------------------------------------------- #
# FGSM / PGD                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("targeted", [False, True])
def test_fgsm_matches_manual(targeted: bool) -> None:
    model = _linear_model()
    x, y = _data()
    sign = -1 if targeted else 1
    expected = (x + sign * 0.05 * _manual_grad(model, x, y).detach().sign()).clamp(0, 1)
    out = wb.fgsm(model, x, y, 0.05, targeted=targeted)
    assert out.shape == x.shape and not out.requires_grad
    assert_close(out, expected)


def test_fgsm_norm_clamp_and_no_side_effects() -> None:
    model = _linear_model()
    x, y = _data()
    x_before = x.clone()
    out = wb.fgsm(model, x, y, 0.1)
    assert (out - x).abs().max().item() == pytest.approx(0.1, abs=1e-6), "||δ||∞ = ε."
    assert torch.equal(x, x_before), "No modifiques la imagen de entrada."
    assert all(p.grad is None for p in model.parameters()), "No acumules gradientes en el modelo."
    extreme = wb.fgsm(model, torch.ones(1, C, H, W), torch.tensor([0]), 0.5)
    assert extreme.max() <= 1 and extreme.min() >= 0, "Recorta a [0, 1]."


def test_fgsm_increases_loss() -> None:
    model = _linear_model()
    x, y = _data()
    adv = wb.fgsm(model, x, y, 0.05)
    assert F.cross_entropy(model(adv), y) > F.cross_entropy(model(x), y)


def test_pgd_one_step_equals_fgsm() -> None:
    model = _linear_model()
    x, y = _data()
    assert_close(wb.pgd_linf(model, x, y, 0.05, 0.05, 1), wb.fgsm(model, x, y, 0.05))


@pytest.mark.parametrize("random_start", [False, True])
def test_pgd_linf_stays_in_ball(random_start: bool) -> None:
    model = DummyCNN(NUM_CLASSES)
    x, y = _data()
    adv = wb.pgd_linf(model, x, y, epsilon=0.03, alpha=0.01, steps=10, random_start=random_start)
    assert (adv - x).abs().max() <= 0.03 + 1e-6, "Proyecta a la bola ℓ∞."
    assert adv.min() >= 0 and adv.max() <= 1


def test_pgd_random_start_depends_on_seed() -> None:
    model = _linear_model()
    x, y = _data()
    torch.manual_seed(0)
    a = wb.pgd_linf(model, x, y, 0.1, 0.0, 1, random_start=True)
    torch.manual_seed(1)
    b = wb.pgd_linf(model, x, y, 0.1, 0.0, 1, random_start=True)
    assert not torch.allclose(a, b), "El inicio aleatorio usa la semilla."


def test_pgd_beats_fgsm_and_targeted_decreases_target_loss() -> None:
    model = _linear_model()
    x, y = _data()
    pgd = wb.pgd_linf(model, x, y, 0.1, 0.02, 20)
    fg = wb.fgsm(model, x, y, 0.1)
    assert F.cross_entropy(model(pgd), y) >= F.cross_entropy(model(fg), y) - 1e-4
    target = (y + 1) % NUM_CLASSES
    adv = wb.pgd_linf(model, x, target, 0.1, 0.02, 20, targeted=True)
    assert F.cross_entropy(model(adv), target) < F.cross_entropy(model(x), target)


def test_pgd_l2() -> None:
    model = _linear_model()
    x, y = _data()
    adv = wb.pgd_l2(model, x, y, epsilon=0.5, alpha=0.2, steps=10)
    norms = (adv - x).flatten(1).norm(dim=1)
    assert (norms <= 0.5 + 1e-4).all(), "Proyecta a la bola ℓ2 por muestra."
    assert F.cross_entropy(model(adv), y) > F.cross_entropy(model(x), y)
    one = wb.pgd_l2(model, x, y, epsilon=10.0, alpha=0.1, steps=1)
    assert_close((one - x).flatten(1).norm(dim=1), torch.full((BATCH,), 0.1), atol=1e-4)


# --------------------------------------------------------------------------- #
# I-FGSM (estructura del examen)                                              #
# --------------------------------------------------------------------------- #


def test_ifgsm_init() -> None:
    model = _linear_model()
    img = torch.rand(C, H, W)
    attack = wb.IFGSMAttack(img, model, true_class_idx=1)
    assert torch.equal(attack.perturbed_img, attack.img), "Empieza siendo una copia."
    assert attack.perturbed_img is not attack.img
    assert isinstance(attack.loss_fn, nn.CrossEntropyLoss)
    assert not model.training, "El modelo pasa a eval."
    with pytest.raises(ValueError):
        wb.IFGSMAttack(img, model, 1, targeted=True)


def test_ifgsm_compute_loss() -> None:
    model = _linear_model()
    img = torch.rand(C, H, W)
    untargeted = wb.IFGSMAttack(img, model, 1)
    targeted = wb.IFGSMAttack(img, model, 1, targeted=True, target_class_idx=2)
    logits = model(img.unsqueeze(0))
    assert_close(untargeted._compute_loss(img), F.cross_entropy(logits, torch.tensor([1])))
    assert_close(targeted._compute_loss(img), F.cross_entropy(logits, torch.tensor([2])))


def test_ifgsm_gradient_step_equals_fgsm() -> None:
    model = _linear_model()
    img = torch.rand(C, H, W) * 0.6 + 0.2
    attack = wb.IFGSMAttack(img, model, 1)
    attack._gradient_step(epsilon=0.1, alpha=0.02)
    expected = wb.fgsm(model, img.unsqueeze(0), torch.tensor([1]), 0.02)[0]
    assert_close(attack.perturbed_img, expected)
    assert len(attack.loss_history) == 1


def test_ifgsm_fit_projection_and_norms() -> None:
    model = _linear_model()
    img = torch.rand(C, H, W) * 0.6 + 0.2
    attack = wb.IFGSMAttack(img, model, 0)
    attack.fit(epochs=20, epsilon=0.05, alpha=0.01)
    assert len(attack.loss_history) == 20
    assert attack.get_norm("linf") <= 0.05 + 1e-6
    delta = attack.get_perturbation()
    assert attack.get_norm("l2") == pytest.approx(delta.norm().item(), rel=1e-5)
    assert attack.get_norm("l1") == pytest.approx(delta.abs().sum().item(), rel=1e-5)
    with pytest.raises(ValueError):
        attack.get_norm("l3")


@pytest.mark.parametrize("targeted", [False, True])
def test_ifgsm_success(targeted: bool) -> None:
    model = _linear_model()
    img = torch.rand(C, H, W) * 0.6 + 0.2
    true = int(model(img.unsqueeze(0)).argmax())
    target = (true + 1) % NUM_CLASSES
    attack = wb.IFGSMAttack(img, model, true, targeted=targeted, target_class_idx=target)
    assert not attack.get_attack_success(), "Sin atacar no hay éxito."
    attack.fit(epochs=100, epsilon=0.5, alpha=0.02)
    assert attack.get_attack_success()
    pred = int(model(attack.perturbed_img.unsqueeze(0)).argmax())
    assert (pred == target) if targeted else (pred != true)


# --------------------------------------------------------------------------- #
# C&W y ejemplo lineal                                                        #
# --------------------------------------------------------------------------- #


def test_carlini_wagner_reaches_target() -> None:
    model = _linear_model()
    x, _ = _data()
    target = (model(x).argmax(1) + 1) % NUM_CLASSES
    adv = wb.carlini_wagner_l2(model, x, target, c=10.0, kappa=0.1, steps=200, lr=0.05)
    assert adv.shape == x.shape and not adv.requires_grad
    assert adv.min() >= 0 and adv.max() <= 1, "tanh mantiene x' en [0, 1]."
    assert (model(adv).argmax(1) == target).all(), "Alcanza la clase objetivo."


def test_carlini_wagner_zero_steps_is_identity() -> None:
    model = _linear_model()
    x, y = _data()
    assert_close(wb.carlini_wagner_l2(model, x, y, steps=0), x, atol=1e-4)


def test_linear_worst_case() -> None:
    weight = torch.tensor([0.5, -2.0, 1.0, 0.0])
    delta, change = wb.linear_worst_case(weight, 0.1)
    assert_close(delta, torch.tensor([0.1, -0.1, 0.1, 0.0]))
    assert change == pytest.approx(0.1 * 3.5)
    _, big = wb.linear_worst_case(torch.ones(1000), 1 / 255)
    assert big == pytest.approx(1000 / 255), "Crece con la dimensión: n / 255."


# --------------------------------------------------------------------------- #
# One Pixel Attack                                                            #
# --------------------------------------------------------------------------- #


def test_apply_pixel_perturbations() -> None:
    img = torch.zeros(3, 4, 5)
    pert = torch.tensor([[0.5, 0.0, 1.0, 0.5, 0.25], [1.0, 1.0, 0.1, 0.2, 0.3]])
    out = bb.apply_pixel_perturbations(img, pert)
    assert torch.equal(img, torch.zeros(3, 4, 5)), "Trabaja sobre una copia."
    assert_close(out[:, 2, 0], torch.tensor([1.0, 0.5, 0.25]))  # fila floor(0.5·4)=2
    assert_close(out[:, 3, 4], torch.tensor([0.1, 0.2, 0.3]))  # x = y = 1 → última fila/columna
    assert (out != 0).any(dim=0).sum() == 2, "Solo cambian dos píxeles."
    clipped = bb.apply_pixel_perturbations(img, torch.tensor([[-1.0, 2.0, 3.0, -1.0, 0.5]]))
    assert_close(clipped[:, 0, 4], torch.tensor([1.0, 0.0, 0.5]))


def test_one_pixel_evaluate_matches_softmax() -> None:
    model = DummyCNN(NUM_CLASSES)
    img = torch.rand(3, H, W)
    attack = bb.OnePixelAttack(model, img, label=1, population_size=6)
    assert attack.fitness.shape == (6,), "Un fitness por individuo."
    i = 3
    expected = F.softmax(model(bb.apply_pixel_perturbations(img, attack.population[i : i + 1])[None]), 1)[0, 1]
    assert_close(attack.fitness[i], expected, atol=1e-6)


def test_one_pixel_step_never_worsens() -> None:
    model = DummyCNN(NUM_CLASSES)
    attack = bb.OnePixelAttack(model, torch.rand(3, H, W), label=0, population_size=8)
    for _ in range(5):
        before = attack.fitness.clone()
        attack.step()
        assert (attack.fitness <= before).all(), "La selección nunca empeora a un individuo."
        assert attack.population.min() >= 0 and attack.population.max() <= 1
    assert len(attack.history) == 6
    assert all(a >= b for a, b in zip(attack.history, attack.history[1:]))


def test_one_pixel_attack_output() -> None:
    model = DummyCNN(NUM_CLASSES)
    img = torch.rand(3, H, W)
    attack = bb.OnePixelAttack(model, img, label=0, population_size=8)
    adv, best = attack.attack(generations=3)
    assert adv.shape == img.shape and best.shape == (5,)
    assert (adv != img).any(dim=0).sum() <= 1, "Solo cambia un píxel."
    prob = F.softmax(model(adv[None]), 1)[0, 0]
    assert_close(prob, attack.fitness.min(), atol=1e-6)


# --------------------------------------------------------------------------- #
# Defensas                                                                    #
# --------------------------------------------------------------------------- #


def test_robust_accuracy() -> None:
    model = _linear_model()
    x, _ = _data()
    y = model(x).argmax(1)
    clean, robust = d.robust_accuracy(model, x, y, lambda m, a, b: wb.fgsm(m, a, b, 0.5))
    assert clean == 1.0
    assert robust < clean
    same = d.robust_accuracy(model, x, y, lambda m, a, b: a)
    assert same == (1.0, 1.0)


def test_transfer_success_rate() -> None:
    model = _linear_model()
    x, _ = _data()
    y = model(x).argmax(1)
    attack = lambda m, a, b: wb.pgd_linf(m, a, b, 0.5, 0.05, 30)  # noqa: E731
    assert d.transfer_success_rate(model, model, x, y, attack) == 1.0, "White-box: transfiere al 100%."
    assert d.transfer_success_rate(model, model, x, y, lambda m, a, b: a) == 0.0
    wrong = (y + 1) % NUM_CLASSES
    assert d.transfer_success_rate(model, model, x, wrong, lambda m, a, b: a) == 0.0


def test_adversarial_training_step() -> None:
    model = _linear_model()
    x, y = _data()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [p.clone() for p in model.parameters()]
    adv = wb.fgsm(model, x, y, 0.1)
    expected = 0.5 * F.cross_entropy(model(x), y) + 0.5 * F.cross_entropy(model(adv), y)
    loss = d.adversarial_training_step(model, optimizer, x, y, 0.1)
    assert loss == pytest.approx(expected.item(), rel=1e-5)
    assert model.training, "Termina en modo train."
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters())), "Actualiza pesos."
    fresh = _linear_model()
    expected_clean = F.cross_entropy(fresh(x), y).item()
    clean = d.adversarial_training_step(fresh, torch.optim.SGD(fresh.parameters(), lr=0.1), x, y, 0.1, 0.0)
    assert clean == pytest.approx(expected_clean, rel=1e-5), "Con w = 0 es la CE limpia."


def test_vat_loss() -> None:
    model = _linear_model()
    x, _ = _data()
    loss = d.vat_loss(model, x, epsilon=0.5)
    assert loss.dim() == 0 and loss.item() >= 0
    loss.backward()
    assert model[1].weight.grad is not None, "Es derivable respecto al modelo."
    small = d.vat_loss(model, x, epsilon=0.01).item()
    assert small < loss.item(), "Con ε menor la KL es menor."


def test_vat_finds_adversarial_direction() -> None:
    """La dirección de la iteración de potencia da más KL que una aleatoria."""

    model = _linear_model()
    x, _ = _data()
    torch.manual_seed(3)
    vat = d.vat_loss(model, x, epsilon=0.5, xi=1e-3).item()
    with torch.no_grad():
        p = F.softmax(model(x), 1)
        rand = d._l2_normalize(torch.randn_like(x)) * 0.5
        random_kl = F.kl_div(F.log_softmax(model(x + rand), 1), p, reduction="batchmean").item()
    assert vat > random_kl
