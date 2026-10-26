"""
Tests de F04 · Funciones de activación.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from helpers import assert_close, load
from torch import nn

m = load("fundamentals/04_activations/activations.py")

EXTREME = torch.tensor([-1000.0, -50.0, -1.0, 0.0, 1.0, 50.0, 1000.0])
FORBIDDEN = (
    "F.",
    "nn.functional",
    "torch.relu(",
    "torch.sigmoid(",
    "torch.tanh(",
    "torch.softmax(",
)

# (nombre, implementación propia, referencia de PyTorch)
ELEMENTWISE: list[tuple[str, Callable, Callable]] = [
    ("relu", m.relu, F.relu),
    ("leaky_relu", m.leaky_relu, F.leaky_relu),
    ("leaky_relu_0.2", lambda x: m.leaky_relu(x, 0.2), lambda x: F.leaky_relu(x, 0.2)),
    ("elu", m.elu, F.elu),
    ("elu_0.5", lambda x: m.elu(x, 0.5), lambda x: F.elu(x, 0.5)),
    ("sigmoid", m.sigmoid, torch.sigmoid),
    ("tanh", m.tanh, torch.tanh),
    ("gelu", m.gelu, F.gelu),
    ("gelu_tanh", lambda x: m.gelu(x, "tanh"), lambda x: F.gelu(x, approximate="tanh")),
    ("silu", m.silu, F.silu),
    ("softplus", m.softplus, F.softplus),
    ("softplus_b2", lambda x: m.softplus(x, 2.0), lambda x: F.softplus(x, beta=2.0)),
    ("mish", m.mish, F.mish),
    ("hardsigmoid", m.hardsigmoid, F.hardsigmoid),
    ("hardswish", m.hardswish, F.hardswish),
]
IDS = [name for name, _, _ in ELEMENTWISE]


# ---------------------------------------------------------------------------
# Funciones elemento a elemento
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,fn,ref", ELEMENTWISE, ids=IDS)
@pytest.mark.parametrize("shape", [(7,), (2, 3, 4, 4)])
def test_matches_pytorch(
    name: str, fn: Callable, ref: Callable, shape: tuple[int, ...]
) -> None:
    """Coincide con PyTorch en valores normales y conserva forma y dtype."""

    x = 3 * torch.randn(shape)
    out = fn(x)
    assert out.shape == x.shape and out.dtype == x.dtype, (
        f"{name}: forma o dtype incorrectos."
    )
    assert_close(out, ref(x))


@pytest.mark.parametrize("name,fn,ref", ELEMENTWISE, ids=IDS)
def test_extreme_values(name: str, fn: Callable, ref: Callable) -> None:
    """Con ±1000 no hay inf/nan ni en la salida ni en el gradiente."""

    x = EXTREME.clone().requires_grad_(True)
    out = fn(x)
    assert torch.isfinite(out).all(), f"{name}: salida no finita."
    assert_close(out.detach(), ref(EXTREME), atol=1e-4)
    out.sum().backward()
    assert torch.isfinite(x.grad).all(), (
        f"{name}: gradiente no finito (¿rama con exp(1000)?)."
    )


@pytest.mark.parametrize("name,fn,ref", ELEMENTWISE, ids=IDS)
def test_gradients_match_pytorch(name: str, fn: Callable, ref: Callable) -> None:
    """El gradiente que da autograd sobre tu implementación coincide con el de PyTorch."""

    x = (3 * torch.randn(20)).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    fn(x).sum().backward()
    ref(x_ref).sum().backward()
    assert_close(x.grad, x_ref.grad, atol=1e-4)


def test_hand_values() -> None:
    """Valores conocidos calculados a mano."""

    assert torch.equal(
        m.relu(torch.tensor([-2.0, 0.0, 3.0])), torch.tensor([0.0, 0.0, 3.0])
    )
    assert torch.isclose(m.leaky_relu(torch.tensor(-10.0), 0.2), torch.tensor(-2.0))
    assert torch.isclose(m.sigmoid(torch.tensor(0.0)), torch.tensor(0.5))
    assert torch.isclose(m.softplus(torch.tensor(0.0)), torch.log(torch.tensor(2.0)))
    assert torch.isclose(m.gelu(torch.tensor(0.0)), torch.tensor(0.0))
    assert torch.isclose(m.hardswish(torch.tensor(3.0)), torch.tensor(3.0))
    assert torch.isclose(m.hardsigmoid(torch.tensor(-3.0)), torch.tensor(0.0))
    assert torch.isclose(m.elu(torch.tensor(-1000.0)), torch.tensor(-1.0)), (
        "ELU satura en -α."
    )


def test_ranges() -> None:
    """Rangos de salida."""

    x = 10 * torch.randn(1000)
    s = m.sigmoid(x)
    assert ((s >= 0) & (s <= 1)).all(), "sigmoid ∈ [0, 1]."
    t = m.tanh(x)
    assert ((t >= -1) & (t <= 1)).all(), "tanh ∈ [-1, 1]."
    assert (m.softplus(x) > 0).all(), "softplus > 0."
    assert (m.elu(x) >= -1).all(), "ELU(α=1) ≥ -1."
    assert m.silu(x).min() >= -0.2785, "El mínimo de SiLU es ≈ -0.2785."


def test_gelu_invalid() -> None:
    """``approximate`` desconocido → ValueError."""

    with pytest.raises(ValueError):
        m.gelu(torch.randn(3), approximate="erf")


def test_sigmoid_derivative() -> None:
    """σ'(0) = 1/4, coincide con autograd y es finita en ±1000."""

    assert torch.isclose(m.sigmoid_derivative(torch.tensor(0.0)), torch.tensor(0.25))
    x = torch.randn(10, requires_grad=True)
    torch.sigmoid(x).sum().backward()
    assert_close(m.sigmoid_derivative(x.detach()), x.grad)
    d = m.sigmoid_derivative(EXTREME)
    assert torch.isfinite(d).all() and (d >= 0).all() and d.max() <= 0.25, (
        "σ' ∈ [0, 1/4]."
    )


# ---------------------------------------------------------------------------
# Softmax y log-softmax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_softmax(dim: int) -> None:
    """Coincide con ``torch.softmax`` y suma 1 en el eje indicado."""

    x = 5 * torch.randn(3, 4, 5)
    out = m.softmax(x, dim=dim)
    assert_close(out, torch.softmax(x, dim=dim))
    assert_close(out.sum(dim=dim), torch.ones_like(out.sum(dim=dim)))


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_log_softmax(dim: int) -> None:
    """Coincide con ``torch.log_softmax``."""

    x = 5 * torch.randn(3, 4, 5)
    assert_close(m.log_softmax(x, dim=dim), torch.log_softmax(x, dim=dim))


def test_softmax_extreme_logits() -> None:
    """Logits ±1000: sin nan/inf y gradientes finitos (hay que restar el máximo)."""

    x = torch.tensor(
        [[1000.0, 0.0, -1000.0], [1000.0, 1000.0, -1000.0]], requires_grad=True
    )
    p = m.softmax(x, dim=1)
    assert torch.isfinite(p).all(), "softmax no finita: resta el máximo."
    assert_close(p.detach(), torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]))

    lp = m.log_softmax(x, dim=1)
    assert torch.isfinite(lp).all(), "log_softmax no finita: no hagas log(softmax)."
    assert_close(lp.detach(), torch.log_softmax(x.detach(), dim=1))
    assert torch.isclose(lp[0, 2], torch.tensor(-2000.0)), "log p = -2000."

    (lp[:, 0].sum() + p[:, 1].sum()).backward()
    assert torch.isfinite(x.grad).all(), "Gradiente no finito."


def test_log_softmax_gives_cross_entropy() -> None:
    """NLL sobre tu log_softmax = ``F.cross_entropy`` sobre los logits."""

    logits, labels = torch.randn(6, 10), torch.randint(0, 10, (6,))
    ce = -m.log_softmax(logits, dim=1).gather(1, labels[:, None]).mean()
    assert torch.isclose(ce, F.cross_entropy(logits, labels), atol=1e-5)


# ---------------------------------------------------------------------------
# PReLU
# ---------------------------------------------------------------------------


def test_prelu_parameter() -> None:
    """Un parámetro aprendible con el valor inicial pedido."""

    act = m.PReLU()
    params = list(act.parameters())
    assert len(params) == 1 and params[0].shape == (1,), (
        "Un único parámetro de forma (1,)."
    )
    assert torch.isclose(params[0], torch.tensor([0.25])).all(), "init = 0.25."
    assert torch.equal(act(torch.tensor([-4.0, 2.0])), torch.tensor([-1.0, 2.0]))


@pytest.mark.parametrize("num_parameters", [1, 3])
def test_prelu_matches_nn(num_parameters: int) -> None:
    """Misma salida y gradiente que ``nn.PReLU`` (también por canal)."""

    mine, ref = m.PReLU(num_parameters, init=0.1), nn.PReLU(num_parameters, init=0.1)
    x = torch.randn(2, 3, 4, 4)
    out_m, out_r = mine(x), ref(x)
    assert out_m.shape == x.shape, "Forma incorrecta."
    assert_close(out_m, out_r)
    out_m.sum().backward()
    out_r.sum().backward()
    assert_close(mine.weight.grad, ref.weight.grad)


def test_prelu_learns() -> None:
    """La pendiente cambia tras un paso de optimización."""

    act = m.PReLU(3)
    old = act.weight.detach().clone()
    opt = torch.optim.SGD(act.parameters(), lr=0.1)
    act(-torch.rand(4, 3, 2, 2)).sum().backward()
    opt.step()
    assert torch.norm(act.weight - old) > 0, "La pendiente debe actualizarse."


# ---------------------------------------------------------------------------
# Desde cero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn_name",
    [
        "relu",
        "leaky_relu",
        "elu",
        "sigmoid",
        "tanh",
        "gelu",
        "silu",
        "softplus",
        "mish",
        "hardsigmoid",
        "hardswish",
        "softmax",
        "log_softmax",
    ],
)
def test_from_scratch(fn_name: str) -> None:
    """No se usan las activaciones ya hechas de PyTorch."""

    src = inspect.getsource(getattr(m, fn_name))
    for token in FORBIDDEN:
        assert token not in src, (
            f"{fn_name}: no uses {token!r}, impleméntala desde cero."
        )
