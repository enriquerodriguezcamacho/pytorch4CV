"""
Tests de CV05 · Explicabilidad: utilidades y métodos de gradiente.
"""

import pytest
import torch
from helpers import DummyCNN, assert_close, load
from torch import nn

u = load("cv/05_explainability/utils.py")
g = load("cv/05_explainability/gradients.py")

BATCH = 2
CHANNELS = 3
HEIGHT = 5
WIDTH = 6
NUM_CLASSES = 4


def linear_model(seed: int = 0) -> nn.Module:
    """Modelo lineal ``Flatten → Linear``: el gradiente es la fila de pesos."""

    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Flatten(), nn.Linear(CHANNELS * HEIGHT * WIDTH, NUM_CLASSES)
    )


def class_weights(model: nn.Module, c: int) -> torch.Tensor:
    """Pesos de la clase ``c`` con forma de imagen [C, H, W]."""

    return model[1].weight[c].detach().view(CHANNELS, HEIGHT, WIDTH)


class QuadraticModel(nn.Module):
    """$f(x) = \\sum x^2$ (un solo logit); su gradiente es $2x$."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x**2).flatten(1).sum(dim=1, keepdim=True)


def tiny_relu_mlp() -> nn.Module:
    """MLP 2 → 3 → 1 con pesos fijos para calcular Guided Backprop a mano."""

    model = nn.Sequential(
        nn.Flatten(), nn.Linear(2, 3, bias=False), nn.ReLU(), nn.Linear(3, 1, bias=False)
    )
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[1.0, -1.0], [2.0, 1.0], [1.0, 1.0]]))
        model[3].weight.copy_(torch.tensor([[1.0, 2.0, -1.0]]))
    return model


# Entrada del MLP: h = [-1, 4, 3] → ReLU → [0, 4, 3]
MLP_INPUT = torch.tensor([1.0, 2.0]).view(1, 1, 1, 2)
VANILLA_GRAD = torch.tensor([3.0, 1.0])  # W1ᵀ · ([1, 2, -1] ⊙ [0, 1, 1])
GUIDED_GRAD = torch.tensor([4.0, 2.0])  # W1ᵀ · clamp([0, 2, -1], 0)
DECONV_GRAD = torch.tensor([5.0, 1.0])  # W1ᵀ · clamp([1, 2, -1], 0)


# ----------------------------------------------------------------- utils ---- #


def test_normalize_map_range_and_per_sample() -> None:
    """Cada muestra se normaliza por separado a [0, 1]."""

    x = torch.stack([torch.rand(HEIGHT, WIDTH), 100 * torch.rand(HEIGHT, WIDTH) + 5])
    out = u.normalize_map(x)
    assert out.shape == x.shape, "La forma no debe cambiar."
    for i in range(2):
        assert torch.isclose(out[i].min(), torch.tensor(0.0), atol=1e-6), "Mín ≠ 0."
        assert torch.isclose(out[i].max(), torch.tensor(1.0), atol=1e-6), "Máx ≠ 1."


def test_normalize_map_hand_values_and_constant() -> None:
    """Valores calculados a mano, entrada 2D y mapa constante → ceros."""

    out = u.normalize_map(torch.tensor([[2.0, 4.0], [6.0, 10.0]]))
    assert_close(out, torch.tensor([[0.0, 0.25], [0.5, 1.0]]), atol=1e-6)
    const = u.normalize_map(torch.full((1, 3, 3), 7.0))
    assert torch.all(const == 0), "Un mapa constante debe quedar a cero."
    assert torch.isfinite(const).all(), "No debe haber NaN."


def test_select_target() -> None:
    """``None`` → argmax; entero → constante; fuera de rango → ValueError."""

    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 0.5]])
    assert torch.equal(u.select_target(logits, None), torch.tensor([1, 0]))
    out = u.select_target(logits, 2)
    assert out.dtype == torch.int64, "Los índices deben ser int64."
    assert torch.equal(out, torch.tensor([2, 2])), "Clase fija incorrecta."
    with pytest.raises(ValueError):
        u.select_target(logits, 3)


# ------------------------------------------------------------- gradients ---- #


def test_input_gradient_linear_and_no_param_grads() -> None:
    """En un modelo lineal el gradiente es la fila de pesos; sin grads en params."""

    model = linear_model()
    x = torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH)
    grad = g.input_gradient(model, x, target_class=2)
    assert grad.shape == x.shape, "Forma incorrecta."
    for i in range(BATCH):
        assert_close(grad[i], class_weights(model, 2))
    assert all(p.grad is None for p in model.parameters()), (
        "No se deben acumular gradientes en los parámetros (usa autograd.grad)."
    )
    assert not x.requires_grad, "La entrada original no debe modificarse."


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("target_class", [None, 1])
def test_saliency_map_shape_and_sign(batch: int, target_class: int | None) -> None:
    """Forma [batch, H, W] y valores no negativos con DummyCNN."""

    model = DummyCNN(num_classes=NUM_CLASSES)
    x = torch.randn(batch, CHANNELS, HEIGHT, WIDTH)
    out = g.saliency_map(model, x, target_class)
    assert out.shape == (batch, HEIGHT, WIDTH), "Forma incorrecta."
    assert torch.all(out >= 0), "La saliencia es un valor absoluto."


def test_saliency_map_linear_hand_value() -> None:
    """Saliency = max sobre canales de |w_c|."""

    model = linear_model()
    x = torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH)
    out = g.saliency_map(model, x, target_class=0)
    expected = class_weights(model, 0).abs().amax(dim=0)
    assert_close(out[0], expected)
    assert_close(out[1], expected)


def test_saliency_map_uses_predicted_class() -> None:
    """Con ``target_class=None`` explica la clase predicha de cada muestra."""

    model = linear_model()
    x = torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH)
    preds = model(x).argmax(dim=1)
    out = g.saliency_map(model, x, None)
    for i in range(BATCH):
        expected = class_weights(model, int(preds[i])).abs().amax(dim=0)
        assert_close(out[i], expected)


def test_input_x_gradient_hand_value() -> None:
    """Input × Gradient = max_k |x · w|."""

    model = linear_model()
    x = torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH)
    out = g.input_x_gradient(model, x, target_class=3)
    assert out.shape == (BATCH, HEIGHT, WIDTH), "Forma incorrecta."
    assert_close(out, (x * class_weights(model, 3)).abs().amax(dim=1))


def test_input_x_gradient_zero_input() -> None:
    """Una imagen negra tiene atribución nula (penaliza píxeles de valor bajo)."""

    model = DummyCNN(num_classes=NUM_CLASSES)
    out = g.input_x_gradient(model, torch.zeros(1, CHANNELS, HEIGHT, WIDTH), 0)
    assert torch.all(out == 0), "x = 0 ⇒ x · grad = 0."


def test_smoothgrad_without_noise_equals_saliency() -> None:
    """``noise_level=0`` reproduce exactamente el saliency map."""

    model = DummyCNN(num_classes=NUM_CLASSES)
    x = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    out = g.smoothgrad(model, x, 1, n_samples=3, noise_level=0.0)
    assert_close(out, g.saliency_map(model, x, 1))


def test_smoothgrad_linear_model_is_noise_invariant() -> None:
    """En un modelo lineal el gradiente no depende del punto: SmoothGrad = saliency."""

    model = linear_model()
    x = torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH)
    out = g.smoothgrad(model, x, 2, n_samples=5, noise_level=0.5)
    assert_close(out, g.saliency_map(model, x, 2))


def test_smoothgrad_deterministic_with_seed() -> None:
    """Misma semilla ⇒ mismo resultado; semillas distintas ⇒ distinto."""

    model = DummyCNN(num_classes=NUM_CLASSES)
    x = torch.randn(1, CHANNELS, HEIGHT, WIDTH)
    torch.manual_seed(0)
    a = g.smoothgrad(model, x, 0, n_samples=4, noise_level=0.2)
    torch.manual_seed(0)
    b = g.smoothgrad(model, x, 0, n_samples=4, noise_level=0.2)
    torch.manual_seed(1)
    c = g.smoothgrad(model, x, 0, n_samples=4, noise_level=0.2)
    assert a.shape == (1, HEIGHT, WIDTH), "Forma incorrecta."
    assert torch.allclose(a, b), "Debe ser determinista con la misma semilla."
    assert not torch.allclose(a, c), "Semillas distintas deben dar ruido distinto."


# --------------------------------------------------------- integrated grads -- #


@pytest.mark.parametrize("n_steps", [1, 7, 50])
def test_integrated_gradients_completeness_linear(n_steps: int) -> None:
    """Modelo lineal: ΣIG = f(x) − f(0) exactamente, para cualquier n_steps."""

    model = linear_model()
    x = torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH)
    attr = g.integrated_gradients(model, x, target_class=1, n_steps=n_steps)
    assert attr.shape == x.shape, "IG devuelve atribuciones por canal."
    with torch.no_grad():
        diff = model(x)[:, 1] - model(torch.zeros_like(x))[:, 1]
    assert_close(attr.flatten(1).sum(dim=1), diff, atol=1e-4)


@pytest.mark.parametrize("n_steps", [1, 4, 10])
def test_integrated_gradients_quadratic_hand_value(n_steps: int) -> None:
    """f = Σx²: IG = x · (1/M) Σ_k 2(k/M)x = x² (M + 1) / M."""

    x = torch.tensor([[[[1.0, -2.0], [0.5, 3.0]]]])
    attr = g.integrated_gradients(QuadraticModel(), x, target_class=0, n_steps=n_steps)
    assert_close(attr, x**2 * (n_steps + 1) / n_steps)


def test_integrated_gradients_random_baseline() -> None:
    """Baseline aleatorio ``torch.rand_like``: completitud respecto a ese baseline."""

    model = linear_model()
    x = torch.rand(1, CHANNELS, HEIGHT, WIDTH)
    torch.manual_seed(3)
    attr = g.integrated_gradients(model, x, 0, baseline="random", n_steps=5)
    torch.manual_seed(3)
    x_base = torch.rand_like(x)
    with torch.no_grad():
        diff = model(x)[0, 0] - model(x_base)[0, 0]
    assert torch.isclose(attr.sum(), diff, atol=1e-4), "Falla la completitud."
    zero_attr = g.integrated_gradients(model, x, 0, baseline="zero", n_steps=5)
    assert not torch.allclose(attr, zero_attr), "El baseline debe influir."


def test_integrated_gradients_completeness_nonlinear() -> None:
    """Con una CNN con ReLU la completitud se cumple aproximadamente (M grande)."""

    model = DummyCNN(num_classes=NUM_CLASSES)
    x = torch.rand(1, CHANNELS, 8, 8)
    attr = g.integrated_gradients(model, x, target_class=2, n_steps=300)
    with torch.no_grad():
        diff = model(x)[0, 2] - model(torch.zeros_like(x))[0, 2]
    assert torch.isclose(attr.sum(), diff, atol=2e-3, rtol=0.05), (
        "ΣIG debería aproximar f(x) − f(x′)."
    )


def test_integrated_gradients_invalid_baseline() -> None:
    """Un baseline desconocido lanza ValueError."""

    with pytest.raises(ValueError):
        g.integrated_gradients(linear_model(), torch.rand(1, 3, 5, 6), 0, "blur")


# ------------------------------------------------------- guided backprop ---- #


def test_vanilla_gradient_tiny_mlp() -> None:
    """Referencia: backprop normal en el MLP de pesos fijos."""

    grad = g.input_gradient(tiny_relu_mlp(), MLP_INPUT, 0)
    assert_close(grad.flatten(), VANILLA_GRAD)


def test_guided_backprop_hand_value() -> None:
    """Guided Backprop anula el gradiente negativo que llega a la ReLU."""

    explainer = g.GuidedBackprop(tiny_relu_mlp())
    grad = explainer.attribute(MLP_INPUT, 0)
    assert grad.shape == MLP_INPUT.shape, "Forma incorrecta."
    assert_close(grad.flatten(), GUIDED_GRAD)


def test_deconvnet_hand_value() -> None:
    """DeconvNet ignora la máscara del forward (la unidad con h < 0 también pasa)."""

    assert_close(g.deconvnet(tiny_relu_mlp(), MLP_INPUT, 0).flatten(), DECONV_GRAD)
    assert_close(
        g.DeconvNet(tiny_relu_mlp()).attribute(MLP_INPUT, 0).flatten(), DECONV_GRAD
    )


def test_guided_backprop_register_and_remove_hooks() -> None:
    """Un forward y un backward hook por ReLU; se eliminan tras ``attribute``."""

    model = DummyCNN(num_classes=NUM_CLASSES)
    explainer = g.GuidedBackprop(model)
    assert len(explainer.handles) == 0, "No debe haber hooks al construir."
    explainer._register_hooks()
    n_relus = sum(isinstance(m, nn.ReLU) for m in model.modules())
    assert len(explainer.handles) == 2 * n_relus, "Un hook forward + backward por ReLU."
    model(torch.randn(1, CHANNELS, HEIGHT, WIDTH))
    assert len(explainer.activations) == n_relus, "Se deben guardar las activaciones."
    explainer._remove_hooks()
    assert len(explainer.handles) == 0 and len(explainer.activations) == 0, (
        "Hooks/activaciones no eliminados."
    )


@pytest.mark.parametrize("target_class", [None, 1])
def test_guided_backprop_explain(target_class: int | None) -> None:
    """``explain`` devuelve un mapa [B, H, W] en [0, 1] y limpia los hooks."""

    explainer = g.GuidedBackprop(DummyCNN(num_classes=NUM_CLASSES))
    out = explainer.explain(torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH), target_class)
    assert out.shape == (BATCH, HEIGHT, WIDTH), "Forma incorrecta."
    assert out.min() >= 0 and out.max() <= 1, "Debe estar normalizado."
    assert len(explainer.handles) == 0, "Los hooks deben eliminarse."


def test_guided_backprop_without_relu_raises() -> None:
    """Un modelo sin ``nn.ReLU`` lanza RuntimeError."""

    explainer = g.GuidedBackprop(linear_model())
    with pytest.raises(RuntimeError):
        explainer.attribute(torch.rand(1, CHANNELS, HEIGHT, WIDTH), 0)
