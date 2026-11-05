"""
Tests de los ejercicios del tema 5 de los apuntes.
"""

import pytest
import torch
from helpers import DummyCNN, assert_close, load
from torch import nn

a = load("cv/05_explainability/apuntes.py")
p = load("cv/05_explainability/perturbation.py")


def test_monte_carlo_shapley_additive_game() -> None:
    """En un juego aditivo basta una permutación para el valor exacto."""

    weights = [1.0, 2.0, 5.0]
    values = a.monte_carlo_shapley(lambda s: sum(weights[i] for i in s), 3, 1)
    assert_close(values, torch.tensor(weights))


def test_monte_carlo_shapley_converges_to_exact() -> None:
    """Con muchas permutaciones se acerca al valor exacto y cumple eficiencia."""

    def value_fn(coalition: frozenset[int]) -> float:
        return float(len(coalition) ** 2)

    estimate = a.monte_carlo_shapley(value_fn, 4, 300)
    assert_close(estimate, p.exact_shapley_values(value_fn, 4), atol=0.25)
    assert_close(estimate.sum(), torch.tensor(16.0))


def test_shapley_pixel_map_shape() -> None:
    """Un valor por celda repetido en sus píxeles."""

    result = a.shapley_pixel_map(DummyCNN().eval(), torch.rand(3, 8, 8), 0, 2, 2)
    assert result.shape == (8, 8)
    assert torch.equal(result[:4, :4], result[0, 0].expand(4, 4))


def test_loyalty_scores_definition() -> None:
    """Empieza en 1 (nada eliminado), está acotada y ``inverse`` cambia el orden."""

    model, x = DummyCNN().eval(), torch.rand(3, 8, 8)
    saliency = torch.rand(8, 8)
    curve, area = a.loyalty_scores(model, x, saliency, 0, n_steps=4)
    inverse, _ = a.loyalty_scores(model, x, saliency, 0, inverse=True, n_steps=4)
    assert curve.shape == (5,) and isinstance(area, float)
    assert_close(curve[0], torch.tensor(1.0))
    assert not torch.equal(curve[1:-1], inverse[1:-1])


def test_remove_most_important() -> None:
    """Pone a ``fill`` justo los píxeles con mayor importancia, sin tocar la entrada."""

    images = torch.ones(1, 3, 2, 2)
    saliency = torch.tensor([[[0.1, 0.9], [0.8, 0.2]]])
    out = a.remove_most_important(images, saliency, 0.5)
    expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).expand(1, 3, 2, 2)
    assert_close(out, expected)
    assert images.min() == 1


def test_roar_accuracy_uses_fresh_model() -> None:
    """Entrena modelos nuevos: aprende el dato limpio y cae al quitar la señal."""

    x = torch.randn(64, 3, 2, 2)
    y = (x[:, 0, 0, 0] > 0).long()
    created = []

    def factory() -> nn.Module:
        created.append(1)
        return nn.Sequential(nn.Flatten(), nn.Linear(12, 2))

    clean = a.roar_accuracy(factory, (x, y), (x, y), epochs=100, lr=0.1)
    removed = x.clone()
    removed[:, 0, 0, 0] = 0
    corrupted = a.roar_accuracy(factory, (removed, y), (removed, y), epochs=100, lr=0.1)
    assert len(created) == 2
    assert clean > 0.9 and corrupted < clean


def test_cascading_randomization_is_cumulative() -> None:
    """Cada copia acumula las capas ya aleatorizadas y el original no cambia."""

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    original = [param.data.clone() for param in model.parameters()]
    first, second = a.cascading_randomization(model, ["1", "0"])
    assert not torch.equal(first[1].weight, model[1].weight)
    assert torch.equal(first[0].weight, model[0].weight)
    assert not torch.equal(second[0].weight, model[0].weight)
    assert torch.equal(second[1].weight, first[1].weight)
    assert all(
        torch.equal(old, new.data) for old, new in zip(original, model.parameters())
    )


def test_shuffle_labels_is_permutation() -> None:
    """Conserva las etiquetas y cambia su orden."""

    labels = torch.arange(50)
    shuffled = a.shuffle_labels(labels)
    assert torch.equal(shuffled.sort().values, labels)
    assert not torch.equal(shuffled, labels)


def test_global_ssim_values() -> None:
    """Vale 1 con mapas iguales y menos si se invierte el mapa."""

    x = torch.rand(8, 8)
    assert_close(a.global_ssim(x, x), torch.tensor(1.0))
    assert a.global_ssim(x, 1 - x) < 0.5
    assert a.global_ssim(x, x + 0.01) == pytest.approx(1.0, abs=0.01)
