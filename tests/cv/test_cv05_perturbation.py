"""
Tests de CV05 · Explicabilidad: Occlusion, valores de Shapley y RISE.
"""

import math

import pytest
import torch
from helpers import DummyCNN, assert_close, load
from torch import nn

p = load("cv/05_explainability/perturbation.py")

HEIGHT = 8
WIDTH = 8


def _linear_model(weight: torch.Tensor) -> nn.Module:
    """Modelo lineal sin sesgo: logits = W · vec(x)."""

    model = nn.Sequential(nn.Flatten(), nn.Linear(weight.shape[1], weight.shape[0], bias=False))
    with torch.no_grad():
        model[1].weight.copy_(weight)
    return model


def test_occlusion_linear_model_pixelwise() -> None:
    """Con patch=1 y modelo lineal, el mapa es |Σ_c w_c(i, j) · x_c(i, j)|."""

    x = torch.randn(1, 2, 3, 3)
    weight = torch.randn(2, 18)
    out = p.occlusion_map(_linear_model(weight), x, target_class=1, patch=1, stride=1)
    expected = (weight[1].view(2, 3, 3) * x[0]).sum(0).abs()
    assert out.shape == (1, 3, 3), "Forma incorrecta."
    assert_close(out[0], expected)


def test_occlusion_seen_average() -> None:
    """Parche 2, stride 1: cada píxel promedia las diferencias de los parches que lo tapan."""

    x = torch.ones(1, 1, 3, 3)
    model = _linear_model(torch.ones(1, 9))  # logit = suma de píxeles
    out = p.occlusion_map(model, x, target_class=0, patch=2, stride=1)
    # el parche de esquina (i, j) tapa r(i)·r(j) píxeles, con r = (2, 2, 1) por el recorte
    expected = torch.tensor([[4.0, 4.0, 3.0], [4.0, 4.0, 3.0], [3.0, 3.0, 2.25]])
    assert_close(out[0], expected)


def _strong_model() -> nn.Module:
    """Modelo lineal 3 clases cuya probabilidad varía claramente con cada máscara.

    Con la DummyCNN la salida casi no cambia entre máscaras y la normalización
    min-max amplifica el ruido de redondeo entre la versión en bucle y por lotes.
    """

    return _linear_model(torch.randn(3, 3 * HEIGHT * WIDTH) * 0.3)


def test_occlusion_shape_batch_and_no_grad() -> None:
    model = DummyCNN()
    x = torch.randn(2, 3, HEIGHT, WIDTH)
    out = p.occlusion_map(model, x, patch=4, stride=2)
    assert out.shape == (2, HEIGHT, WIDTH), "Forma incorrecta."
    assert not out.requires_grad, "Occlusion no necesita gradientes."
    assert (out >= 0).all(), "Se acumulan diferencias en valor absoluto."


def test_exact_shapley_additive_game() -> None:
    """En un juego aditivo v(S) = Σ_{i∈S} a_i, φ_i = a_i."""

    a = [1.0, -2.0, 0.5, 3.0]
    phi = p.exact_shapley_values(lambda s: sum(a[i] for i in s), len(a))
    assert phi.shape == (4,), "Forma incorrecta."
    assert_close(phi, torch.tensor(a))


def test_exact_shapley_interaction_and_efficiency() -> None:
    """v(S) = 1 solo si 0 y 1 están en S: se reparte a medias. Σφ = v(N) − v(∅)."""

    phi = p.exact_shapley_values(lambda s: float({0, 1} <= s), 3)
    assert_close(phi, torch.tensor([0.5, 0.5, 0.0]))

    def value(s: frozenset[int]) -> float:
        return len(s) ** 2 + (2.0 if 2 in s else 0.0)

    phi = p.exact_shapley_values(value, 4)
    assert phi.sum().item() == pytest.approx(value(frozenset(range(4))) - value(frozenset()))


def test_kernel_shap_weight() -> None:
    assert p.kernel_shap_weight(4, 1) == pytest.approx(3 / (4 * 1 * 3))
    assert p.kernel_shap_weight(4, 2) == pytest.approx(3 / (6 * 2 * 2))
    assert p.kernel_shap_weight(4, 0) == math.inf
    assert p.kernel_shap_weight(4, 4) == math.inf
    with pytest.raises(ValueError):
        p.kernel_shap_weight(4, 5)


def test_percentage_validation() -> None:
    assert p.Percentage(0.3).value == 0.3
    with pytest.raises(ValueError):
        p.Percentage(1.5)


def test_rise_select_mask_function() -> None:
    model = DummyCNN()
    assert p.RISE(model, "naive").mask_function == p.BaseRISE.create_mask_naive
    assert p.RISE(model, "normal").mask_function == p.BaseRISE.create_mask
    with pytest.raises(ValueError):
        p.RISE(model, "gaussian")


def test_create_mask_naive_binary() -> None:
    mask = p.BaseRISE.create_mask_naive(p.Percentage(0.5), 32, 32)
    assert mask.shape == (1, 32, 32), "Forma incorrecta."
    assert set(mask.unique().tolist()) <= {0.0, 1.0}, "Debe ser binaria."
    assert p.BaseRISE.create_mask_naive(p.Percentage(1.0), 4, 4).eq(1).all()


def test_create_mask_smooth_and_shape() -> None:
    mask = p.BaseRISE.create_mask(p.Percentage(0.5), 20, 30, s=4)
    assert mask.shape == (1, 20, 30), "Forma incorrecta."
    assert (mask >= 0).all() and (mask <= 1).all(), "Valores en [0, 1]."
    ones = p.BaseRISE.create_mask(p.Percentage(1.0), 20, 30, s=4)
    assert_close(ones, torch.ones(1, 20, 30))
    smooth = p.BaseRISE.create_mask(p.Percentage(0.5), 64, 64, s=4)
    assert ((smooth > 0) & (smooth < 1)).any(), "El upsampling bilineal da valores intermedios."


@pytest.mark.parametrize("mask_method", ["naive", "normal"])
def test_rise_output_range(mask_method: str) -> None:
    model = _strong_model()
    out = p.RISE(model, mask_method).explain(torch.rand(3, HEIGHT, WIDTH), n_steps=20, s=2)
    assert out.shape == (HEIGHT, WIDTH), "Forma incorrecta."
    assert out.min().item() == pytest.approx(0.0, abs=1e-5)
    assert out.max().item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("batch_size", [1, 4, 7])
def test_vectorized_rise_matches_loop(batch_size: int) -> None:
    """Misma semilla → mismo resultado, aunque n_steps no sea múltiplo del lote."""

    model = _strong_model()
    x = torch.rand(3, HEIGHT, WIDTH)
    torch.manual_seed(0)
    loop = p.RISE(model).explain(x, n_steps=18, s=3, target_class=1)
    torch.manual_seed(0)
    vec = p.VectorizedRISE(model).explain(x, batch_size=batch_size, n_steps=18, s=3, target_class=1)
    assert_close(vec, loop, atol=1e-4)


def test_rise_seed_changes_result() -> None:
    model = _strong_model()
    x = torch.rand(3, HEIGHT, WIDTH)
    torch.manual_seed(0)
    a = p.RISE(model).explain(x, n_steps=10, s=3)
    torch.manual_seed(1)
    b = p.RISE(model).explain(x, n_steps=10, s=3)
    assert not torch.allclose(a, b), "Otra semilla debe dar otras máscaras."
