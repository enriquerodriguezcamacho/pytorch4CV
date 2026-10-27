"""
Tests de CV05 · Explicabilidad: evaluación de explicaciones y visualización de features.
"""

import pytest
import torch
from helpers import DummyCNN, assert_close, load
from torch import nn

e = load("cv/05_explainability/evaluation.py")
fv = load("cv/05_explainability/feature_viz.py")


def _sum_model() -> nn.Module:
    """Logit de la clase 0 = suma de píxeles; clase 1 = 0."""

    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2, bias=False))
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]))
    return model


# --------------------------------------------------------------------------- #
# Evaluación                                                                  #
# --------------------------------------------------------------------------- #


def test_auc() -> None:
    assert e.auc(torch.full((5,), 0.7)) == pytest.approx(0.7)
    assert e.auc(torch.linspace(0, 1, 11)) == pytest.approx(0.5)
    assert e.auc(torch.tensor([0.0, 1.0, 0.0])) == pytest.approx(0.5)


def test_deletion_curve_hand_values() -> None:
    """Imagen de unos 2×2, 4 pasos: el logit baja de 4 a 0 de uno en uno."""

    x = torch.ones(1, 2, 2)
    saliency = torch.rand(2, 2)
    curve = e.perturbation_curve(_sum_model(), x, saliency, 0, "deletion", n_steps=4, use_softmax=False)
    assert curve.shape == (5,), "Debe tener n_steps + 1 puntos."
    assert_close(curve, torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]))


def test_insertion_curve_and_order() -> None:
    """Insertar primero el píxel más importante (valor 3)."""

    x = torch.tensor([[[3.0, 1.0], [0.0, 2.0]]])
    saliency = x[0].clone()
    most = e.perturbation_curve(_sum_model(), x, saliency, 0, "insertion", "most", 4, use_softmax=False)
    least = e.perturbation_curve(_sum_model(), x, saliency, 0, "insertion", "least", 4, use_softmax=False)
    assert_close(most, torch.tensor([0.0, 3.0, 5.0, 6.0, 6.0]))
    assert_close(least, torch.tensor([0.0, 0.0, 1.0, 3.0, 6.0]))


def test_perturbation_curve_softmax_and_rounding() -> None:
    """Con softmax son probabilidades; ceil(H·W / n_steps) píxeles por paso."""

    model = DummyCNN()
    x = torch.rand(3, 5, 5)
    curve = e.perturbation_curve(model, x, torch.rand(5, 5), 1, n_steps=3)
    assert curve.shape == (4,)
    assert ((curve >= 0) & (curve <= 1)).all(), "Con use_softmax son probabilidades."
    full = torch.softmax(model(torch.zeros(1, 3, 5, 5)), dim=1)[0, 1]
    assert_close(curve[-1], full, atol=1e-5)  # 3 · ceil(25/3) = 27 >= 25: todo borrado


def test_perturbation_curve_invalid() -> None:
    with pytest.raises(ValueError):
        e.perturbation_curve(_sum_model(), torch.ones(1, 2, 2), torch.rand(2, 2), 0, "blur")
    with pytest.raises(ValueError):
        e.perturbation_curve(_sum_model(), torch.ones(1, 2, 2), torch.rand(2, 2), 0, order="random")


def test_good_explanation_has_lower_deletion() -> None:
    """La saliencia correcta borra antes los píxeles importantes: AUC de deletion menor."""

    x = torch.tensor([[[3.0, 1.0], [0.0, 2.0]]])
    good = e.deletion_score(_sum_model(), x, x[0], 0, 4, use_softmax=False)
    bad = e.deletion_score(_sum_model(), x, -x[0], 0, 4, use_softmax=False)
    assert good < bad
    assert e.insertion_score(_sum_model(), x, x[0], 0, 4, False) > e.insertion_score(
        _sum_model(), x, -x[0], 0, 4, False
    )


def test_rank_correlation() -> None:
    a = torch.randn(6, 6)
    assert e.rank_correlation(a, a) == pytest.approx(1.0)
    assert e.rank_correlation(a, a.exp() * 3) == pytest.approx(1.0), "Solo importan los rangos."
    assert e.rank_correlation(a, -a) == pytest.approx(-1.0)


def test_randomize_layers() -> None:
    model = DummyCNN()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    new = e.randomize_layers(model, ["fc", "features.4"])
    assert new is not model, "Devuelve una copia."
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), "El modelo original no se modifica."
    assert not torch.equal(new.fc.weight, model.fc.weight)
    assert not torch.equal(new.features[4].weight, model.features[4].weight)
    assert torch.equal(new.features[0].weight, model.features[0].weight), "Las demás capas no cambian."
    with pytest.raises(KeyError):
        e.randomize_layers(model, ["no_existe"])


# --------------------------------------------------------------------------- #
# Visualización de features                                                   #
# --------------------------------------------------------------------------- #


def test_normalize_filters() -> None:
    conv = nn.Conv2d(3, 4, 5)
    out = fv.normalize_filters(conv)
    assert out.shape == (4, 3, 5, 5), "Forma incorrecta."
    assert not out.requires_grad, "Copia sin gradiente."
    assert_close(out.amin(dim=(1, 2, 3)), torch.zeros(4))
    assert_close(out.amax(dim=(1, 2, 3)), torch.ones(4), atol=1e-5)


def test_max_activating_indices() -> None:
    acts = torch.zeros(4, 2, 3, 3)
    acts[0, 1, 0, 0] = 5.0
    acts[2, 1, 2, 2] = 9.0
    acts[3, 1] = 1.0  # media alta pero máximo bajo
    acts[1, 0, 1, 1] = 100.0  # otro canal
    assert fv.max_activating_indices(acts, 1, 3).tolist() == [2, 0, 3]


def test_gradient_ascent_linear_model() -> None:
    """Con S_c(I) = wᵀI: el primer paso desde cero es I = lr · w y no se tocan los pesos."""

    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    weights_before = model[1].weight.clone()
    image, history = fv.gradient_ascent_class_visualization(
        model, 1, (1, 1, 2, 2), n_steps=1, lr=0.5, l2_reg=0.1
    )
    assert image.shape == (1, 1, 2, 2), "Forma incorrecta."
    assert not image.requires_grad
    assert_close(image.flatten(), 0.5 * model[1].weight[1].detach())
    assert len(history) == 1
    assert history[0] == pytest.approx(model[1].bias[1].item(), abs=1e-6)
    assert torch.equal(model[1].weight, weights_before), "No modifiques el modelo."


def test_gradient_ascent_objective_increases() -> None:
    model = DummyCNN()
    _, history = fv.gradient_ascent_class_visualization(
        model, 0, (1, 3, 8, 8), n_steps=20, lr=0.05, l2_reg=1e-2
    )
    assert len(history) == 20
    assert history[-1] > history[0], "El objetivo debe subir."


def test_feature_inversion_identity_converges() -> None:
    target = torch.rand(1, 3, 4, 4)
    image, history = fv.feature_inversion(lambda x: x, target, n_steps=300, lr=0.05)
    assert image.shape == target.shape and not image.requires_grad
    assert len(history) == 300
    assert history[-1] < history[0] * 1e-2, "La pérdida debe bajar."
    assert_close(image, target, atol=5e-2)


def test_deep_dream() -> None:
    image = torch.rand(1, 3, 6, 6) * 0.5 + 0.25
    original = image.clone()
    out = fv.deep_dream(lambda x: x, image, n_steps=5, lr=0.05)
    assert out.shape == image.shape and not out.requires_grad
    assert torch.equal(image, original), "No modifiques la imagen de entrada."
    assert ((out >= 0) & (out <= 1)).all(), "Recorta a [0, 1]."
    assert (out**2).mean() > (image**2).mean(), "Amplifica las activaciones."
