"""
Tests de F12 · Hooks y extracción de características.
"""

import copy

import pytest
import torch
import torch.nn.functional as F
from helpers import DummyCNN, assert_close, load
from torch import nn

m = load("fundamentals/12_hooks_feature_extraction/hooks.py")

BATCH = 2
IMAGE_SIZE = 8
NUM_CLASSES = 3
CONV_NAMES = ["features.0", "features.2", "features.4"]


@pytest.fixture
def x() -> torch.Tensor:
    """Lote de imágenes pequeñas."""

    return torch.rand(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)


def num_hooks(model: nn.Module, kind: str = "forward") -> int:
    """Cuenta los hooks registrados en todos los submódulos."""

    attr = {"forward": "_forward_hooks", "pre": "_forward_pre_hooks"}.get(
        kind, "_backward_hooks"
    )
    return sum(len(getattr(mod, attr)) for mod in model.modules())


# ---------------------------------------------------------------------------
# CustomModel
# ---------------------------------------------------------------------------
def test_custom_model_num_hooks() -> None:
    """Un hook por cada ``nn.Conv2d``."""

    base = DummyCNN(NUM_CLASSES)
    model = m.CustomModel(base)
    n_convs = sum(isinstance(mod, nn.Conv2d) for mod in base.modules())
    assert len(model.hooks) == n_convs, "Debe haber un hook por Conv2d."
    assert num_hooks(base) == n_convs, "Los hooks deben estar en las Conv2d."
    for mod in base.modules():
        if isinstance(mod, nn.Conv2d):
            assert len(mod._forward_hooks) == 1, "Cada Conv2d debe tener 1 hook."


def test_custom_model_forward_collects_all(x: torch.Tensor) -> None:
    """Un forward deja una activación por convolución con la forma correcta."""

    model = m.CustomModel(DummyCNN(NUM_CLASSES))
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES), "El forward debe ser el del modelo."
    assert len(model.activations) == 3, "Debe haber 3 activaciones."
    assert model.activations[0].shape == (BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.mark.parametrize("layers", [[0], [1, 2], [0, 1, 2]])
@pytest.mark.parametrize("use_grad", [True, False])
def test_obtain_activations(x: torch.Tensor, layers: list[int], use_grad: bool) -> None:
    """Claves pedidas, valores correctos y ``requires_grad`` según ``use_grad``."""

    base = DummyCNN(NUM_CLASSES)
    model = m.CustomModel(base)
    acts = model.obtain_activations(x, layers, use_grad)
    assert isinstance(acts, dict), "Debe devolver un diccionario."
    assert sorted(acts) == sorted(layers), "Claves incorrectas."
    for value in acts.values():
        assert value.requires_grad == use_grad, "requires_grad no coincide."
    with torch.no_grad():
        expected = base.features[:1](x)
    if 0 in acts:
        assert_close(acts[0].detach(), expected)


def test_obtain_activations_clears_previous(x: torch.Tensor) -> None:
    """Llamadas sucesivas no acumulan activaciones."""

    model = m.CustomModel(DummyCNN(NUM_CLASSES))
    model.obtain_activations(x, [0], False)
    model.obtain_activations(torch.zeros_like(x), [0], False)
    assert len(model.activations) == 3, "Hay que vaciar activations antes."
    with torch.no_grad():
        expected = model.model.features[0](torch.zeros_like(x))
    assert_close(model.activations[0], expected)


def test_obtain_activations_gradient_flows(x: torch.Tensor) -> None:
    """Con ``use_grad=True`` se puede hacer backward hasta la imagen."""

    model = m.CustomModel(DummyCNN(NUM_CLASSES))
    img = x.clone().requires_grad_()
    acts = model.obtain_activations(img, [2], True)
    acts[2].pow(2).sum().backward()
    assert img.grad is not None and img.grad.abs().sum() > 0, "No llega gradiente."


def test_remove_hooks_and_activations(x: torch.Tensor) -> None:
    """``_remove_hooks_and_activations`` limpia listas y módulos."""

    base = DummyCNN(NUM_CLASSES)
    model = m.CustomModel(base)
    model(x)
    model._remove_hooks_and_activations()
    assert model.hooks == [] and model.activations == [], "Las listas deben vaciarse."
    assert num_hooks(base) == 0, "Quedan hooks registrados."
    model(x)
    assert model.activations == [], "Sin hooks no se deben guardar activaciones."


# ---------------------------------------------------------------------------
# register_activation_hooks y count_activation_shapes
# ---------------------------------------------------------------------------
def test_register_activation_hooks_late_binding(x: torch.Tensor) -> None:
    """Cada nombre guarda **su** activación (detecta el bug de late binding)."""

    model = DummyCNN(NUM_CLASSES)
    acts, handles = m.register_activation_hooks(model, ["features.0", "features.3"])
    assert acts == {} and len(handles) == 2, "Diccionario vacío y 2 handles."
    model(x)
    assert set(acts) == {"features.0", "features.3"}, "Claves incorrectas."
    with torch.no_grad():
        assert_close(acts["features.0"], model.features[:1](x))
        assert_close(acts["features.3"], model.features[:4](x))
    assert not acts["features.0"].requires_grad, "Hay que hacer detach()."
    for h in handles:
        h.remove()
    assert num_hooks(model) == 0, "remove() debe quitar los hooks."


def test_register_activation_hooks_raises() -> None:
    """Un nombre inexistente → ``KeyError``."""

    with pytest.raises(KeyError):
        m.register_activation_hooks(DummyCNN(), ["no_existe"])


def test_count_activation_shapes(x: torch.Tensor) -> None:
    """Formas de las Conv2d y de la Linear; el modelo queda sin hooks."""

    model = DummyCNN(NUM_CLASSES)
    shapes = m.count_activation_shapes(model, x)
    assert list(shapes) == CONV_NAMES, "Claves/orden incorrectos."
    assert all(s == (BATCH, 3, IMAGE_SIZE, IMAGE_SIZE) for s in shapes.values())
    assert m.count_activation_shapes(model, x, nn.Linear) == {"fc": (BATCH, 3)}
    assert num_hooks(model) == 0, "Los hooks temporales deben quitarse."


def test_count_activation_shapes_removes_on_error() -> None:
    """Aunque el forward falle, los hooks se quitan (``try/finally``)."""

    model = DummyCNN(NUM_CLASSES)
    with pytest.raises(RuntimeError):
        m.count_activation_shapes(model, torch.rand(1, 5, 4, 4))
    assert num_hooks(model) == 0, "Quedan hooks tras un error."


# ---------------------------------------------------------------------------
# forward pre-hook
# ---------------------------------------------------------------------------
def test_input_normalization_pre_hook(x: torch.Tensor) -> None:
    """El pre-hook normaliza la entrada y ``remove()`` lo deshace."""

    model = DummyCNN(NUM_CLASSES)
    mean, std = (0.5, 0.4, 0.3), (0.2, 0.25, 0.5)
    with torch.no_grad():
        expected = model((x - torch.tensor(mean).view(1, 3, 1, 1)) / 0.0 + 0.0) * 0
        normalized = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(
            std
        ).view(1, 3, 1, 1)
        expected = model(normalized)
        plain = model(x)
    handle = m.add_input_normalization_hook(model, mean, std)
    assert num_hooks(model, "pre") == 1, "Debe registrarse un forward pre-hook."
    with torch.no_grad():
        assert_close(model(x), expected)
    handle.remove()
    with torch.no_grad():
        assert_close(model(x), plain)


# ---------------------------------------------------------------------------
# GradientCapture y Grad-CAM
# ---------------------------------------------------------------------------
def test_gradient_capture(x: torch.Tensor) -> None:
    """Captura activación y gradiente respecto a la salida de la capa."""

    model = DummyCNN(NUM_CLASSES)
    layer = model.features[2]
    cap = m.GradientCapture(layer)
    assert len(cap.handles) == 2, "Deben registrarse 2 hooks."
    logits = model(x)
    logits[:, 0].sum().backward()

    # referencia con autograd.grad sobre la activación intermedia
    a = model.features[:3](x)
    ref_grad = torch.autograd.grad(model.fc(model.pool(model.features[3:](a)).flatten(1))[:, 0].sum(), a)[0]
    assert cap.activations is not None and cap.gradients is not None, "Vacíos."
    assert not cap.activations.requires_grad, "Hay que hacer detach()."
    assert_close(cap.activations, a.detach())
    assert_close(cap.gradients, ref_grad)
    cap.remove()
    assert num_hooks(model) == 0 and num_hooks(model, "backward") == 0, "Quedan hooks."


def test_grad_cam_shape_range_and_value(x: torch.Tensor) -> None:
    """Forma, rango [0, 1], hooks quitados y valor frente a un cálculo manual."""

    model = DummyCNN(NUM_CLASSES)
    target = torch.tensor([0, 2])
    cam = m.grad_cam(model, model.features[4], x, target)
    assert cam.shape == (BATCH, IMAGE_SIZE, IMAGE_SIZE), "Forma incorrecta."
    assert cam.min() >= 0 and cam.max() <= 1 + 1e-6, "Debe estar en [0, 1]."
    assert num_hooks(model) == 0 and num_hooks(model, "backward") == 0, "Quedan hooks."

    a = model.features[:5](x)
    logits = model.fc(model.pool(model.features[5:](a)).flatten(1))
    g = torch.autograd.grad(logits.gather(1, target[:, None]).sum(), a)[0]
    ref = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1)).detach()
    ref = ref / (ref.amax((1, 2), keepdim=True) + 1e-8)
    assert_close(cam, ref, atol=1e-5)


def test_grad_cam_default_class(x: torch.Tensor) -> None:
    """Sin ``class_idx`` usa la clase predicha."""

    model = DummyCNN(NUM_CLASSES)
    with torch.no_grad():
        pred = model(x).argmax(1)
    assert_close(
        m.grad_cam(model, model.features[2], x),
        m.grad_cam(model, model.features[2], x, pred),
    )


# ---------------------------------------------------------------------------
# Guided backprop
# ---------------------------------------------------------------------------
class _GuidedReLU(torch.autograd.Function):
    """Referencia con ``autograd.Function`` (como en la práctica de XAI)."""

    @staticmethod
    def forward(ctx, inp: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(inp)
        return inp.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        (inp,) = ctx.saved_tensors
        return grad_out.clamp(min=0) * (inp > 0).float()


class _GuidedReLUModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GuidedReLU.apply(x)


def test_guided_backprop_matches_reference(x: torch.Tensor) -> None:
    """Coincide con la implementación con ``autograd.Function``."""

    model = DummyCNN(NUM_CLASSES)
    target = torch.tensor([1, 0])
    out = m.GuidedBackprop(model).attribute(x, target)
    assert out.shape == x.shape, "Forma incorrecta."

    ref_model = copy.deepcopy(model)
    for i, layer in enumerate(ref_model.features):
        if isinstance(layer, nn.ReLU):
            ref_model.features[i] = _GuidedReLUModule()
    inp = x.clone().requires_grad_()
    ref_model(inp).gather(1, target[:, None]).sum().backward()
    assert_close(out, inp.grad)

    plain = x.clone().requires_grad_()
    model(plain).gather(1, target[:, None]).sum().backward()
    assert not torch.allclose(out, plain.grad), "Debe diferir del gradiente normal."


def test_guided_backprop_removes_hooks(x: torch.Tensor) -> None:
    """Tras ``attribute`` no quedan hooks y no se modifica ``x``."""

    model = DummyCNN(NUM_CLASSES)
    gb = m.GuidedBackprop(model)
    x_copy = x.clone()
    gb.attribute(x, torch.tensor([0, 1]))
    assert gb.handles == [], "handles debe quedar vacía."
    assert num_hooks(model, "backward") == 0, "Quedan backward hooks."
    assert not x.requires_grad and torch.equal(x, x_copy), "No modifiques x."


# ---------------------------------------------------------------------------
# create_feature_extractor
# ---------------------------------------------------------------------------
def test_extract_features_fx(x: torch.Tensor) -> None:
    """Coincide con lo capturado mediante hooks."""

    model = DummyCNN(NUM_CLASSES)
    out = m.extract_features_fx(model, {"features.1": "c1", "fc": "logits"}, x)
    assert set(out) == {"c1", "logits"}, "Claves de salida incorrectas."
    with torch.no_grad():
        assert_close(out["c1"], model.features[:2](x))
        assert_close(out["logits"], model(x))
    assert num_hooks(model) == 0, "FX no debe registrar hooks en el modelo."
