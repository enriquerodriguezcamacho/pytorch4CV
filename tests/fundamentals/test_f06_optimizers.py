"""
Tests de F06 · Optimizadores y schedulers.
"""

import math
import warnings

import pytest
import torch
from helpers import DummyCNN, assert_close, load
from torch import nn

m = load("fundamentals/06_optimizers_schedulers/optimizers.py")

NUM_STEPS = 6
LR = 0.05


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))


def _trajectory(model: nn.Module, optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    """Entrena NUM_STEPS pasos con datos fijos y devuelve los pesos tras cada paso."""

    gen = torch.Generator().manual_seed(0)
    x = torch.randn(16, 4, generator=gen)
    y = torch.randn(16, 2, generator=gen)
    out = []
    for _ in range(NUM_STEPS):
        optimizer.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        optimizer.step()
        out.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
    return out


# ----------------------------------------------------------------- SGDMomentum
@pytest.mark.parametrize("momentum", [0.0, 0.9])
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
def test_sgd_momentum_matches_torch(
    momentum: float, nesterov: bool, weight_decay: float
) -> None:
    """La trayectoria coincide con torch.optim.SGD."""

    if nesterov and momentum == 0:
        pytest.skip("Nesterov requiere momentum > 0.")
    model = _tiny_model()
    ref_model = m.clone_model(model)
    ours = m.SGDMomentum(
        model.parameters(),
        lr=LR,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )
    ref = torch.optim.SGD(
        ref_model.parameters(),
        lr=LR,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )
    for a, b in zip(_trajectory(model, ours), _trajectory(ref_model, ref), strict=True):
        assert_close(a, b)


def test_sgd_momentum_hand_value() -> None:
    """f(w) = w²/2 → g = w. w0 = 1, η = 0.1, μ = 0.5: w1 = 0.9, w2 = 0.76."""

    w = nn.Parameter(torch.tensor([1.0]))
    opt = m.SGDMomentum([w], lr=0.1, momentum=0.5)
    expected = [0.9, 0.9 - 0.1 * (0.5 * 1.0 + 0.9)]
    for value in expected:
        opt.zero_grad()
        (0.5 * w**2).sum().backward()
        opt.step()
        assert torch.isclose(w.detach(), torch.tensor([value])).all(), "Paso incorrecto."
    assert "momentum_buffer" in opt.state[w], "Guarda el buffer en self.state[p]."


def test_sgd_momentum_param_groups_and_closure() -> None:
    """Respeta lr por grupo, ignora parámetros sin gradiente y devuelve la closure."""

    a, b, c = (nn.Parameter(torch.ones(2)) for _ in range(3))
    opt = m.SGDMomentum([{"params": [a]}, {"params": [b, c], "lr": 1.0}], lr=0.1, momentum=0)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = (a.sum() + b.sum()) * 1.0
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None and torch.isclose(loss, torch.tensor(4.0)), "Devuelve la pérdida."
    assert_close(a.detach(), torch.full((2,), 0.9))
    assert_close(b.detach(), torch.zeros(2))
    assert_close(c.detach(), torch.ones(2))


@pytest.mark.parametrize("kwargs", [{"lr": -1.0}, {"momentum": 1.0}, {"weight_decay": -0.1}])
def test_sgd_momentum_invalid(kwargs: dict) -> None:
    """Hiperparámetros inválidos → ValueError."""

    with pytest.raises(ValueError):
        m.SGDMomentum([nn.Parameter(torch.ones(1))], **kwargs)


# --------------------------------------------------------------------- Adam
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("decoupled", [False, True])
def test_adam_matches_torch(weight_decay: float, decoupled: bool) -> None:
    """Coincide con torch.optim.Adam (L2) o torch.optim.AdamW (desacoplado)."""

    model = _tiny_model()
    ref_model = m.clone_model(model)
    ours = m.AdamFromScratch(
        model.parameters(),
        lr=1e-2,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled,
    )
    ref_cls = torch.optim.AdamW if decoupled else torch.optim.Adam
    ref = ref_cls(ref_model.parameters(), lr=1e-2, weight_decay=weight_decay)
    for a, b in zip(_trajectory(model, ours), _trajectory(ref_model, ref), strict=True):
        assert_close(a, b)


def test_adam_first_step_is_lr_times_sign() -> None:
    """En el primer paso m̂/√v̂ = sign(g): cada peso se mueve exactamente η."""

    w = nn.Parameter(torch.tensor([3.0, -2.0, 0.5]))
    opt = m.AdamFromScratch([w], lr=0.1)
    opt.zero_grad()
    (w * torch.tensor([5.0, -0.01, 100.0])).sum().backward()
    opt.step()
    assert_close(w.detach(), torch.tensor([2.9, -1.9, 0.4]), atol=1e-6)
    assert opt.state[w]["step"] == 1, "Debe contar los pasos en self.state[p]['step']."


def test_adamw_differs_from_adam_l2() -> None:
    """Con weight decay, AdamW y Adam+L2 dan trayectorias distintas."""

    model = _tiny_model()
    m2 = m.clone_model(model)
    a = _trajectory(model, m.AdamFromScratch(model.parameters(), lr=1e-2, weight_decay=0.5))
    b = _trajectory(
        m2,
        m.AdamFromScratch(m2.parameters(), lr=1e-2, weight_decay=0.5, decoupled_weight_decay=True),
    )
    assert torch.norm(a[-1] - b[-1]) > 1e-4, "L2 y weight decay desacoplado no son iguales."


# --------------------------------------------------------------- param groups
class _NormNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.bn = nn.BatchNorm2d(4)
        self.gn = nn.GroupNorm(2, 4)
        self.ln = nn.LayerNorm(4)
        self.fc = nn.Linear(4, 2, bias=False)
        self.frozen = nn.Linear(2, 2)
        for p in self.frozen.parameters():
            p.requires_grad_(False)


def test_make_param_groups() -> None:
    """Bias y normalizaciones sin decay; congelados fuera."""

    net = _NormNet()
    groups = m.make_param_groups(net, weight_decay=0.05)
    assert len(groups) == 2, "Deben ser dos grupos."
    decay = {id(p) for p in groups[0]["params"]}
    no_decay = {id(p) for p in groups[1]["params"]}
    assert groups[0]["weight_decay"] == 0.05 and groups[1]["weight_decay"] == 0.0
    assert decay == {id(net.conv.weight), id(net.fc.weight)}, "Grupo con decay incorrecto."
    expected_no = {
        id(net.conv.bias),
        *(id(p) for mod in (net.bn, net.gn, net.ln) for p in mod.parameters()),
    }
    assert no_decay == expected_no, "Grupo sin decay incorrecto."
    torch.optim.AdamW(groups, lr=1e-3)  # debe ser aceptado por un optimizador real


def test_make_finetuning_param_groups() -> None:
    """lr distinto para backbone y cabeza; ValueError si no existe la cabeza."""

    net = DummyCNN(num_classes=5)
    groups = m.make_finetuning_param_groups(net, "fc", backbone_lr=1e-4, head_lr=1e-2)
    assert [g["lr"] for g in groups] == [1e-4, 1e-2]
    head = {id(p) for p in groups[1]["params"]}
    assert head == {id(net.fc.weight), id(net.fc.bias)}, "Cabeza incorrecta."
    n_total = sum(1 for _ in net.parameters())
    assert len(groups[0]["params"]) + len(groups[1]["params"]) == n_total
    opt = torch.optim.SGD(groups, lr=0.5)
    assert opt.param_groups[0]["lr"] == 1e-4, "El lr del grupo prevalece sobre el global."
    with pytest.raises(ValueError):
        m.make_finetuning_param_groups(net, "classifier", 1e-4, 1e-2)


# --------------------------------------------------------- warmup + cosine
WARMUP = 4
TOTAL = 12


def test_warmup_cosine_values() -> None:
    """Valores calculados a mano."""

    fn = m.warmup_cosine_lambda(WARMUP, TOTAL, min_factor=0.1)
    assert fn(0) == pytest.approx(0.25)
    assert fn(3) == pytest.approx(1.0)
    assert fn(4) == pytest.approx(1.0)
    assert fn(8) == pytest.approx(0.1 + 0.9 * 0.5)  # mitad del coseno
    assert fn(12) == pytest.approx(0.1)
    assert fn(100) == pytest.approx(0.1), "Tras total_steps se queda en el mínimo."


def test_warmup_cosine_with_lambdalr() -> None:
    """Funciona con LambdaLR y el lr es monótono tras el warmup."""

    w = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([w], lr=0.2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, m.warmup_cosine_lambda(WARMUP, TOTAL))
    lrs = m.run_schedule(opt, sched, TOTAL + 1)
    assert lrs[0] == pytest.approx(0.05) and lrs[WARMUP - 1] == pytest.approx(0.2)
    post = lrs[WARMUP:]
    assert all(a >= b for a, b in zip(post, post[1:])), "El coseno debe decrecer."
    assert lrs[-1] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("warmup, total", [(5, 5), (-1, 10)])
def test_warmup_cosine_invalid(warmup: int, total: int) -> None:
    """Configuraciones imposibles → ValueError."""

    with pytest.raises(ValueError):
        m.warmup_cosine_lambda(warmup, total)


# ---------------------------------------------------------------- schedulers
def test_run_schedule_steplr_and_order() -> None:
    """StepLR: lr se divide por 10 cada 3 pasos; sin warning de orden."""

    w = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([w], lr=1.0)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lrs = m.run_schedule(opt, sched, 7)
    assert_close(torch.tensor(lrs), torch.tensor([1, 1, 1, 0.1, 0.1, 0.1, 0.01]))


def test_run_schedule_cosine() -> None:
    """CosineAnnealingLR sigue la fórmula cerrada."""

    w = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([w], lr=1.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8)
    lrs = m.run_schedule(opt, sched, 9)
    expected = [0.5 * (1 + math.cos(math.pi * t / 8)) for t in range(9)]
    assert_close(torch.tensor(lrs), torch.tensor(expected))


# -------------------------------------------------------------------- EMA
DECAY = 0.9


def test_ema_update_and_copy() -> None:
    """θ_ema = d·θ_ema + (1-d)·θ ; los buffers enteros se copian."""

    model = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2))
    ema = m.ExponentialMovingAverage(model, decay=DECAY)
    w0 = model[0].weight.detach().clone()
    with torch.no_grad():
        model[0].weight.add_(1.0)
    model(torch.randn(4, 3))  # cambia running stats y num_batches_tracked
    ema.update(model)
    assert_close(ema.shadow["0.weight"], DECAY * w0 + (1 - DECAY) * (w0 + 1))
    assert ema.shadow["1.num_batches_tracked"] == 1, "Los buffers enteros se copian."
    target = m.clone_model(model)
    ema.copy_to(target)
    assert_close(target[0].weight.detach(), DECAY * w0 + (1 - DECAY) * (w0 + 1))
    assert torch.norm(model[0].weight - target[0].weight) > 0, "El original no cambia."


def test_ema_independent_copy_and_invalid() -> None:
    """La copia EMA no comparte memoria con el modelo; decay fuera de rango → error."""

    model = nn.Linear(2, 2)
    ema = m.ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.zero_()
    assert torch.norm(ema.shadow["weight"]) > 0, "Debe clonar, no referenciar."
    with pytest.raises(ValueError):
        m.ExponentialMovingAverage(model, decay=1.5)


# -------------------------------------------------------------------- LBFGS
def test_minimize_with_lbfgs_rosenbrock() -> None:
    """Encuentra el mínimo (1, 1) de Rosenbrock sin modificar x0."""

    def rosenbrock(x: torch.Tensor) -> torch.Tensor:
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    x0 = torch.tensor([-1.5, 2.0])
    out = m.minimize_with_lbfgs(rosenbrock, x0, steps=10)
    assert_close(out, torch.tensor([1.0, 1.0]), atol=1e-3)
    assert_close(x0, torch.tensor([-1.5, 2.0]))
    assert not out.requires_grad, "Devuelve un tensor sin gradiente."


# ---------------------------------------------------------- linear regression
TRUE_W = torch.tensor([[2.0, -3.0, 0.5]])
TRUE_B = 1.0


@pytest.mark.parametrize("name, lr", [("sgd", 0.05), ("adam", 0.1), ("adamw", 0.1)])
def test_train_linear_regression(name: str, lr: float) -> None:
    """Recupera los pesos reales y la pérdida baja."""

    x = torch.randn(128, 3)
    y = x @ TRUE_W.T + TRUE_B
    model, history = m.train_linear_regression(x, y, name, lr=lr, epochs=300)
    assert isinstance(model, nn.Linear)
    assert len(history) == 300, "Una pérdida por paso."
    assert history[-1] < 1e-3 < history[0], "La pérdida debe bajar."
    assert_close(model.weight.detach(), TRUE_W, atol=2e-2)
    assert_close(model.bias.detach(), torch.tensor([TRUE_B]), atol=2e-2)


def test_train_linear_regression_invalid() -> None:
    """Optimizador desconocido → ValueError."""

    with pytest.raises(ValueError):
        m.train_linear_regression(torch.randn(4, 1), torch.randn(4, 1), "rmsprop")
