"""
Tests de F03 · nn.Module.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from helpers import DummyCNN, assert_close, load
from torch import nn
from torchvision.models import mobilenet_v3_small, resnet18

m = load("fundamentals/03_nn_module/modules.py")

IN_FEATURES = 6
OUT_FEATURES = 3
BATCH = 5


# ---------------------------------------------------------------------------
# MyLinear
# ---------------------------------------------------------------------------


def test_my_linear_parameters() -> None:
    """Pesos y sesgo registrados como parámetros con la forma correcta."""

    layer = m.MyLinear(IN_FEATURES, OUT_FEATURES)
    names = dict(layer.named_parameters())
    assert set(names) == {"weight", "bias"}, "Registra weight y bias como nn.Parameter."
    assert names["weight"].shape == (OUT_FEATURES, IN_FEATURES), (
        "Forma de weight incorrecta."
    )
    assert names["bias"].shape == (OUT_FEATURES,), "Forma de bias incorrecta."

    no_bias = m.MyLinear(IN_FEATURES, OUT_FEATURES, bias=False)
    assert no_bias.bias is None, "bias debe ser None."
    assert len(list(no_bias.parameters())) == 1, "Sin bias solo hay un parámetro."


@pytest.mark.parametrize("bias", [True, False])
def test_my_linear_matches_nn_linear(bias: bool) -> None:
    """Misma semilla → mismos pesos que ``nn.Linear`` y misma salida."""

    torch.manual_seed(0)
    ref = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=bias)
    torch.manual_seed(0)
    mine = m.MyLinear(IN_FEATURES, OUT_FEATURES, bias=bias)
    assert_close(mine.weight, ref.weight)
    x = torch.randn(2, 4, IN_FEATURES)
    out = mine(x)
    assert out.shape == (2, 4, OUT_FEATURES), "Forma de salida incorrecta."
    assert_close(out, ref(x))


def test_my_linear_trains() -> None:
    """Los parámetros cambian tras un paso de SGD."""

    layer = m.MyLinear(IN_FEATURES, OUT_FEATURES)
    old = layer.weight.detach().clone()
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    layer(torch.randn(BATCH, IN_FEATURES)).pow(2).mean().backward()
    opt.step()
    assert torch.norm(layer.weight - old) > 0, "Los pesos deben actualizarse."


# ---------------------------------------------------------------------------
# MLP / ModuleDict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hidden", [[], [8], [16, 8, 4]])
def test_mlp_structure(hidden: list[int]) -> None:
    """ModuleList con un bloque por capa oculta y número de parámetros correcto."""

    model = m.MLP(IN_FEATURES, hidden, OUT_FEATURES)
    assert isinstance(model.layers, nn.ModuleList), "Usa nn.ModuleList."
    assert len(model.layers) == len(hidden), "Un bloque por capa oculta."
    sizes = [IN_FEATURES, *hidden, OUT_FEATURES]
    expected = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:], strict=True))
    assert sum(p.numel() for p in model.parameters()) == expected, (
        "Nº de parámetros incorrecto."
    )
    out = model(torch.randn(BATCH, IN_FEATURES))
    assert out.shape == (BATCH, OUT_FEATURES), "Forma de salida incorrecta."


def test_mlp_activation_and_dropout() -> None:
    """Se usa la activación pedida y el dropout depende de train/eval."""

    model = m.MLP(IN_FEATURES, [32, 32], OUT_FEATURES, activation=nn.GELU, dropout=0.5)
    assert sum(isinstance(mod, nn.GELU) for mod in model.modules()) == 2, (
        "Una GELU por bloque."
    )
    x = torch.randn(BATCH, IN_FEATURES)
    model.eval()
    assert_close(model(x), model(x))
    model.train()
    assert not torch.allclose(model(x), model(x)), (
        "En train el dropout debe ser aleatorio."
    )


def test_multitask_head() -> None:
    """ModuleDict con una cabeza por tarea."""

    head = m.MultiTaskHead(IN_FEATURES, {"cls": 10, "bbox": 4})
    assert isinstance(head.heads, nn.ModuleDict), "Usa nn.ModuleDict."
    out = head(torch.randn(BATCH, IN_FEATURES))
    assert set(out) == {"cls", "bbox"}, "Claves incorrectas."
    assert out["cls"].shape == (BATCH, 10) and out["bbox"].shape == (BATCH, 4), (
        "Formas incorrectas."
    )
    assert m.count_parameters(head) == IN_FEATURES * 14 + 14, (
        "Parámetros no registrados."
    )


# ---------------------------------------------------------------------------
# Buffers
# ---------------------------------------------------------------------------


def test_running_mean_buffers() -> None:
    """Los buffers están en el state_dict pero no en parameters()."""

    layer = m.RunningMean(3)
    assert len(list(layer.parameters())) == 0, "running_mean no debe ser un parámetro."
    buffers = dict(layer.named_buffers())
    assert set(buffers) == {"running_mean", "num_batches_tracked"}, (
        "Buffers incorrectos."
    )
    assert set(layer.state_dict()) == set(buffers), "Los buffers van en el state_dict."
    assert buffers["num_batches_tracked"].dtype == torch.long, (
        "num_batches_tracked es long."
    )


def test_running_mean_values() -> None:
    """Valores calculados a mano con momentum 0.5."""

    layer = m.RunningMean(2, momentum=0.5)
    x = torch.tensor([[1.0, 2.0], [3.0, 6.0]])  # media [2, 4]
    out = layer(x)
    assert_close(out, torch.tensor([[-1.0, -2.0], [1.0, 2.0]]))
    assert_close(layer.running_mean, torch.tensor([1.0, 2.0]))
    layer(x)
    assert_close(layer.running_mean, torch.tensor([1.5, 3.0]))
    assert layer.num_batches_tracked.item() == 2, "Cuenta los lotes."

    layer.eval()
    out_eval = layer(x)
    assert_close(out_eval, x - torch.tensor([1.5, 3.0]))
    assert layer.num_batches_tracked.item() == 2, "En eval no se actualiza."


def test_running_mean_no_graph_in_buffer() -> None:
    """El buffer no guarda grafo aunque la entrada lo tenga."""

    layer = m.RunningMean(4)
    layer(torch.randn(3, 4, requires_grad=True))
    assert not layer.running_mean.requires_grad, "Actualiza el buffer sin grafo."


# ---------------------------------------------------------------------------
# Contar, congelar, inicializar
# ---------------------------------------------------------------------------


def test_count_parameters() -> None:
    """Cuenta total y entrenable."""

    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))  # 15 + 8
    assert m.count_parameters(model) == 23, "Total incorrecto."
    model[0].requires_grad_(False)
    assert m.count_parameters(model) == 8, "Solo deben contarse los entrenables."
    assert m.count_parameters(model, trainable_only=False) == 23, "Total incorrecto."


@pytest.mark.parametrize(
    "factory,head_name",
    [
        (lambda: DummyCNN(3), "fc"),
        (lambda: resnet18(weights=None, num_classes=5), "fc"),
    ],
)
def test_freeze_all_but_head(factory, head_name: str) -> None:  # noqa: ANN001
    """Solo la cabeza queda entrenable."""

    model = factory()
    out = m.freeze_all_but_head(model, head_name)
    assert out is model, "Devuelve el mismo modelo."
    head = model.get_submodule(head_name)
    head_ids = {id(p) for p in head.parameters()}
    for name, p in model.named_parameters():
        assert p.requires_grad == (id(p) in head_ids), (
            f"{name}: requires_grad incorrecto."
        )
    assert m.count_parameters(model) == sum(p.numel() for p in head.parameters())


def test_init_weights() -> None:
    """apply(init_weights): sesgos a cero, BN a 1/0, varianza de Kaiming fan_out."""

    model = nn.Sequential(
        nn.Conv2d(16, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(10, 20),
    )
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(7.0)
    model.apply(m.init_weights)
    conv, bn, lin = model[0], model[1], model[4]
    assert torch.all(conv.bias == 0) and torch.all(lin.bias == 0), "Sesgos a cero."
    assert torch.all(bn.weight == 1) and torch.all(bn.bias == 0), (
        "BN: peso 1 y sesgo 0."
    )
    expected_std = (2.0 / (64 * 9)) ** 0.5  # gain √2, fan_out = out·k·k
    assert abs(conv.weight.std().item() - expected_std) / expected_std < 0.1, (
        "Usa Kaiming fan_out."
    )
    bound = (6.0 / (10 + 20)) ** 0.5
    assert lin.weight.abs().max() <= bound, "Usa Xavier uniforme."
    assert lin.weight.std() > 0.5 * bound / 3**0.5, (
        "Xavier: varianza demasiado pequeña."
    )


def test_enable_mc_dropout() -> None:
    """Todo en eval salvo los Dropout."""

    model = nn.Sequential(
        nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Dropout(0.5), nn.Linear(8, 2)
    )
    out = m.enable_mc_dropout(model)
    assert out is model, "Devuelve el mismo modelo."
    assert not model.training and not model[1].training, "El modelo y BN en eval."
    assert model[2].training, "Dropout en train."
    x = torch.randn(3, 4)
    assert not torch.allclose(model(x), model(x)), "Dos forwards deben diferir."


# ---------------------------------------------------------------------------
# Guardar y cargar
# ---------------------------------------------------------------------------


def test_save_and_load_weights(tmp_path: Path) -> None:
    """Ida y vuelta del state_dict (incluidos buffers)."""

    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    model(torch.randn(8, 4))  # actualiza running stats
    path = tmp_path / "weights.pt"
    m.save_weights(model, path)
    assert path.exists(), "No se ha guardado el fichero."
    assert isinstance(torch.load(path, weights_only=True), dict), (
        "Guarda el state_dict."
    )

    other = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    out = m.load_weights(other, path)
    assert out is other, "Devuelve el mismo modelo."
    for (k, v1), v2 in zip(
        model.state_dict().items(), other.state_dict().values(), strict=True
    ):
        assert torch.equal(v1, v2), f"{k} no coincide."


def test_load_weights_strict(tmp_path: Path) -> None:
    """``strict=True`` falla con claves distintas y ``strict=False`` no."""

    path = tmp_path / "w.pt"
    m.save_weights(nn.Sequential(nn.Linear(4, 4)), path)
    bigger = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    with pytest.raises(RuntimeError):
        m.load_weights(bigger, path, strict=True)
    m.load_weights(bigger, path, strict=False)


# ---------------------------------------------------------------------------
# Reemplazar la última capa y API funcional
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,attr",
    [
        (lambda: DummyCNN(3), "fc"),
        (lambda: resnet18(weights=None), "fc"),
        (lambda: mobilenet_v3_small(weights=None), "classifier.3"),
    ],
)
@pytest.mark.parametrize("num_classes", [2, 7])
def test_replace_last_layer(factory, attr: str, num_classes: int) -> None:  # noqa: ANN001
    """La última Linear se sustituye y la salida tiene ``num_classes`` logits."""

    model = factory()
    old = model.get_submodule(attr)
    out = m.replace_last_layer(model, num_classes)
    assert out is model, "Devuelve el mismo modelo."
    new = model.get_submodule(attr)
    assert isinstance(new, nn.Linear) and new is not old, "Debe ser una capa nueva."
    assert new.in_features == old.in_features and new.out_features == num_classes
    model.eval()
    with torch.no_grad():
        logits = model(torch.rand(2, 3, 32, 32))
    assert logits.shape == (2, num_classes), "Forma de salida incorrecta."


def test_replace_last_layer_error() -> None:
    """Sin capas lineales → ValueError."""

    with pytest.raises(ValueError):
        m.replace_last_layer(nn.Sequential(nn.Conv2d(3, 3, 1)), 4)


def test_functional_mlp() -> None:
    """Coincide con el MLP equivalente hecho con módulos."""

    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    params = {
        "w1": net[0].weight,
        "b1": net[0].bias,
        "w2": net[2].weight,
        "b2": net[2].bias,
    }
    x = torch.randn(BATCH, 4)
    assert_close(m.functional_mlp(x, params), net(x))
    m.functional_mlp(x, params).sum().backward()
    assert net[0].weight.grad is not None, (
        "Debe propagar gradientes a los tensores dados."
    )
    assert_close(F.relu(torch.tensor([-1.0])), torch.tensor([0.0]))
