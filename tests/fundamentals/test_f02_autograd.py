"""
Tests de F02 · Autograd.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from helpers import DummyCNN, assert_close, load
from torch import nn

m = load("fundamentals/02_autograd/autograd_basics.py")

BATCH = 4
IMAGE_SIZE = 8
NUM_CLASSES = 3
EPSILON = 0.03


# ---------------------------------------------------------------------------
# 1. Derivadas
# ---------------------------------------------------------------------------


def test_df_analytic_hand_values() -> None:
    """f'(0) = -1 y f'(π) = -π² - e^{-π}, calculados a mano."""

    x = torch.tensor([0.0, torch.pi])
    out = m.df_analytic(x)
    expected = torch.tensor([-1.0, -(torch.pi**2) - torch.exp(torch.tensor(-torch.pi))])
    assert torch.isclose(out, expected, atol=1e-5).all(), (
        "Derivada analítica incorrecta."
    )


@pytest.mark.parametrize("n", [1, 7, 50])
def test_df_autograd_matches_analytic(n: int) -> None:
    """Autograd y la derivada analítica coinciden."""

    x = torch.linspace(-3, 3, n)
    assert_close(m.df_autograd(m.f, x), m.df_analytic(x))


def test_df_autograd_does_not_touch_input() -> None:
    """La entrada no se modifica ni se le asigna ``.grad``."""

    x = torch.randn(5)
    out = m.df_autograd(torch.sin, x)
    assert not x.requires_grad and x.grad is None, "No modifiques la entrada."
    assert not out.requires_grad, "El resultado no debe tener grafo."
    assert_close(out, torch.cos(x))


# ---------------------------------------------------------------------------
# 2. Gradiente respecto a la entrada
# ---------------------------------------------------------------------------


def _model_and_data() -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    model = DummyCNN(NUM_CLASSES)
    images = torch.rand(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))
    return model, images, labels


def test_input_gradient() -> None:
    """Forma, valores frente a una referencia y sin gradientes en los parámetros."""

    model, images, labels = _model_and_data()
    model.train()
    grad = m.input_gradient(model, images, labels)
    assert grad.shape == images.shape, "Forma incorrecta."
    assert not model.training, "El modelo debe estar en eval."
    assert all(p.grad is None for p in model.parameters()), (
        "No deben acumularse gradientes en los parámetros."
    )
    assert not images.requires_grad, "No modifiques las imágenes originales."

    x = images.clone().requires_grad_(True)
    F.cross_entropy(model(x), labels).backward()
    assert_close(grad, x.grad)


def test_saliency_map() -> None:
    """Saliencia: forma [B, H, W], no negativa y igual a max_c |∂y_t/∂x|."""

    model, images, labels = _model_and_data()
    sal = m.saliency_map(model, images, labels)
    assert sal.shape == (BATCH, IMAGE_SIZE, IMAGE_SIZE), "Forma incorrecta."
    assert (sal >= 0).all(), "La saliencia debe ser no negativa."

    x = images[:1].clone().requires_grad_(True)
    model(x)[0, labels[0]].backward()
    assert_close(sal[0], x.grad[0].abs().amax(dim=0))


def test_fgsm() -> None:
    """FGSM: perturbación ≤ ε, rango [0, 1] y la pérdida aumenta."""

    model, images, labels = _model_and_data()
    adv = m.fgsm(model, images, labels, EPSILON)
    assert adv.shape == images.shape, "Forma incorrecta."
    assert not adv.requires_grad, "La imagen adversaria no debe tener grafo."
    assert (adv - images).abs().max() <= EPSILON + 1e-6, "Se supera ε."
    assert adv.min() >= 0 and adv.max() <= 1, "Las imágenes deben quedar en [0, 1]."
    with torch.no_grad():
        assert F.cross_entropy(model(adv), labels) > F.cross_entropy(
            model(images), labels
        ), "FGSM debe aumentar la pérdida."


def test_fgsm_zero_epsilon() -> None:
    """Con ε = 0 la imagen no cambia."""

    model, images, labels = _model_and_data()
    assert_close(m.fgsm(model, images, labels, 0.0), images)


# ---------------------------------------------------------------------------
# 3. Backprop manual
# ---------------------------------------------------------------------------


def test_manual_backprop_hand_values() -> None:
    """Ejemplo 1×1 calculado a mano: z = 2·3 + 1 = 7, L = (7 - 5)² = 4."""

    x, w, b, y = (
        torch.tensor([[3.0]]),
        torch.tensor([[2.0]]),
        torch.tensor([1.0]),
        torch.tensor([[5.0]]),
    )
    loss, gw, gb, gx = m.manual_linear_relu_mse(x, w, b, y)
    assert torch.isclose(loss, torch.tensor(4.0)), "Pérdida incorrecta."
    # dL/dz = 2·(7-5) = 4 → dW = 4·3, db = 4, dx = 4·2
    assert torch.isclose(gw, torch.tensor([[12.0]])).all(), "dL/dW incorrecto."
    assert torch.isclose(gb, torch.tensor([4.0])).all(), "dL/db incorrecto."
    assert torch.isclose(gx, torch.tensor([[8.0]])).all(), "dL/dx incorrecto."


@pytest.mark.parametrize("batch", [1, 5])
@pytest.mark.parametrize("in_features,out_features", [(3, 2), (6, 4)])
def test_manual_backprop_matches_autograd(
    batch: int, in_features: int, out_features: int
) -> None:
    """Los gradientes manuales coinciden con los de autograd."""

    x = torch.randn(batch, in_features, requires_grad=True)
    w = torch.randn(out_features, in_features, requires_grad=True)
    b = torch.randn(out_features, requires_grad=True)
    y = torch.randn(batch, out_features)

    with torch.no_grad():
        loss, gw, gb, gx = m.manual_linear_relu_mse(x, w, b, y)

    ref = F.mse_loss(F.relu(F.linear(x, w, b)), y)
    ref.backward()
    assert_close(loss, ref.detach())
    assert_close(gw, w.grad)
    assert_close(gb, b.grad)
    assert_close(gx, x.grad)


# ---------------------------------------------------------------------------
# 4. autograd.Function
# ---------------------------------------------------------------------------


def test_my_sigmoid_gradcheck() -> None:
    """``gradcheck`` (diferencias finitas en float64) valida el backward."""

    x = torch.randn(6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(m.MySigmoid.apply, (x,)), "Backward incorrecto."
    assert_close(m.MySigmoid.apply(x), torch.sigmoid(x))


def test_guided_relu() -> None:
    """Forward = ReLU; backward solo pasa gradientes positivos con entrada positiva."""

    x = torch.tensor([-1.0, 2.0, 3.0, -4.0], requires_grad=True)
    out = m.GuidedReLU.apply(x)
    assert torch.equal(out.detach(), torch.tensor([0.0, 2.0, 3.0, 0.0])), (
        "Forward incorrecto."
    )
    out.backward(torch.tensor([5.0, -6.0, 7.0, 8.0]))
    assert torch.equal(x.grad, torch.tensor([0.0, 0.0, 7.0, 0.0])), (
        "Backward guiado incorrecto."
    )


@pytest.mark.parametrize("lambd", [0.0, 0.5, 2.0])
def test_gradient_reversal(lambd: float) -> None:
    """Identidad en forward y -λ·grad en backward."""

    x = torch.randn(3, 4, requires_grad=True)
    out = m.grad_reverse(x, lambd)
    assert_close(out.detach(), x.detach())
    (out * 3).sum().backward()
    assert_close(x.grad, torch.full((3, 4), -3.0 * lambd))


# ---------------------------------------------------------------------------
# 5. Hooks
# ---------------------------------------------------------------------------


def test_register_gradient_scaler() -> None:
    """El hook escala el gradiente y se puede eliminar."""

    x = torch.ones(3, requires_grad=True)
    handle = m.register_gradient_scaler(x, 10.0)
    (2 * x).sum().backward()
    assert_close(x.grad, torch.full((3,), 20.0))

    handle.remove()
    x.grad = None
    (2 * x).sum().backward()
    assert_close(x.grad, torch.full((3,), 2.0))


# ---------------------------------------------------------------------------
# 6. Segundo orden
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn,d2",
    [
        (lambda x: x**3, lambda x: 6 * x),
        (torch.sin, lambda x: -torch.sin(x)),
        (torch.exp, torch.exp),
    ],
)
def test_second_derivative(fn, d2) -> None:  # noqa: ANN001
    """Segunda derivada frente a la fórmula cerrada."""

    x = torch.linspace(-2, 2, 9)
    assert_close(m.second_derivative(fn, x), d2(x))


def test_gradient_penalty_linear_critic() -> None:
    """Crítico lineal D(x) = w·x: el gradiente es w, así que GP = (‖w‖ - 1)²."""

    critic = nn.Sequential(nn.Flatten(), nn.Linear(3 * 2 * 2, 1, bias=False))
    with torch.no_grad():
        critic[1].weight.fill_(0.5)  # ‖w‖ = 0.5·√12
    real, fake = torch.rand(BATCH, 3, 2, 2), torch.rand(BATCH, 3, 2, 2)
    alpha = torch.rand(BATCH, 1, 1, 1)
    gp = m.gradient_penalty(critic, real, fake, alpha)
    expected = (0.5 * torch.sqrt(torch.tensor(12.0)) - 1) ** 2
    assert gp.shape == (), "Debe ser un escalar."
    assert torch.isclose(gp, expected), "Valor de la penalización incorrecto."


def test_gradient_penalty_is_differentiable() -> None:
    """La penalización propaga gradiente a los parámetros del crítico."""

    critic = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.Tanh(), nn.Flatten(), nn.Linear(4 * 16, 1)
    )
    real, fake = torch.rand(2, 3, 4, 4), torch.rand(2, 3, 4, 4)
    gp = m.gradient_penalty(critic, real, fake, torch.rand(2, 1, 1, 1))
    assert gp.requires_grad, "Usa create_graph=True."
    gp.backward()
    assert critic[0].weight.grad is not None and critic[0].weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 7. Congelar y acumular
# ---------------------------------------------------------------------------


def test_freeze() -> None:
    """Tras congelar, no hay gradientes en los parámetros congelados."""

    model = DummyCNN(NUM_CLASSES)
    out = m.freeze(model.features)
    assert out is model.features, "Debe devolver el mismo módulo."
    assert all(not p.requires_grad for p in model.features.parameters()), (
        "Faltan por congelar."
    )
    assert all(p.requires_grad for p in model.fc.parameters()), "No congeles la cabeza."
    model(torch.rand(2, 3, 8, 8)).sum().backward()
    assert all(p.grad is None for p in model.features.parameters()), (
        "Gradiente en congelados."
    )
    assert model.fc.weight.grad is not None, "La cabeza debe recibir gradiente."


def _linear_setup() -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    x, y = torch.randn(8, 4), torch.randn(8, 2)
    return model, x, y


@pytest.mark.parametrize("steps", [2, 4])
def test_accumulation_equivalent_to_big_batch(steps: int) -> None:
    """Acumular ``steps`` micro-lotes = un paso con el lote completo."""

    model, x, y = _linear_setup()
    big = nn.Linear(4, 2)
    big.load_state_dict(model.state_dict())

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = list(zip(x.chunk(steps), y.chunk(steps), strict=True))
    losses = m.train_with_accumulation(model, F.mse_loss, opt, batches, steps)
    assert len(losses) == steps, "Debe devolver una pérdida por micro-lote."

    opt_big = torch.optim.SGD(big.parameters(), lr=0.1)
    F.mse_loss(big(x), y).backward()
    opt_big.step()

    assert_close(model.weight, big.weight)
    assert_close(model.bias, big.bias)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.parameters()), (
        "Al terminar los gradientes deben quedar a cero."
    )


def test_accumulation_steps_and_errors() -> None:
    """Número de pasos del optimizador (incluido el resto) y ValueError."""

    model, x, y = _linear_setup()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = list(zip(x.chunk(5), y.chunk(5), strict=True))  # 4 micro-lotes de 2
    calls = {"n": 0}
    original = opt.step

    def counting_step(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls["n"] += 1
        return original(*args, **kwargs)

    opt.step = counting_step  # type: ignore[method-assign]
    m.train_with_accumulation(model, F.mse_loss, opt, batches, 3)
    assert calls["n"] == 2, "3 + 1 micro-lotes → 2 pasos del optimizador."
    assert model.training, "El modelo debe estar en train."

    with pytest.raises(ValueError):
        m.train_with_accumulation(model, F.mse_loss, opt, batches, 0)
