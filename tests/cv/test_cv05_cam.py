"""
Tests de CV05 · Explicabilidad: CAM, Grad-CAM, Guided Grad-CAM y attention rollout.
"""

import pytest
import torch
import torch.nn.functional as F
from helpers import DummyCNN, assert_close, load
from torch import nn

c = load("cv/05_explainability/cam.py")
u = load("cv/05_explainability/utils.py")

BATCH = 2
HEIGHT = 8
WIDTH = 10
NUM_CLASSES = 3


class GapNet(nn.Module):
    """Arquitectura tipo CAM: conv + ReLU → GAP → Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.features(x)).flatten(1))


def test_cam_hand_value() -> None:
    """M_c = Σ_k w_ck A_k calculado a mano."""

    features = torch.tensor(
        [[[[1.0, 0.0], [2.0, 1.0]], [[0.0, 3.0], [1.0, -1.0]]]]
    )  # [1, 2, 2, 2]
    fc_weight = torch.tensor([[1.0, 1.0], [2.0, -1.0], [0.0, 0.5]])
    out = c.cam(features, fc_weight, torch.tensor([1]))
    expected = torch.tensor([[[2.0, -3.0], [3.0, 3.0]]])
    assert out.shape == (1, 2, 2), "Forma incorrecta."
    assert_close(out, expected)


def test_cam_per_sample_targets() -> None:
    """Cada muestra del lote usa los pesos de su propia clase."""

    features = torch.randn(BATCH, 4, 3, 3)
    fc_weight = torch.randn(NUM_CLASSES, 4)
    targets = torch.tensor([2, 0])
    out = c.cam(features, fc_weight, targets)
    for i in range(BATCH):
        expected = (fc_weight[targets[i]].view(4, 1, 1) * features[i]).sum(dim=0)
        assert_close(out[i], expected)


@pytest.mark.parametrize("target_class", [0, 2])
def test_gradcam_equals_cam_with_gap(target_class: int) -> None:
    """Con conv → GAP → Linear, Grad-CAM coincide con CAM (tras ReLU y normalizar)."""

    model = GapNet()
    x = torch.randn(BATCH, 3, HEIGHT, WIDTH)
    explainer = c.GradCAM(model, model.features)
    out = explainer.generate(x, target_class)
    with torch.no_grad():
        feats = model.features(x)
    targets = torch.full((BATCH,), target_class)
    expected = u.normalize_map(F.relu(c.cam(feats, model.fc.weight.detach(), targets)))
    assert out.shape == (BATCH, HEIGHT, WIDTH), "Forma incorrecta."
    assert_close(out, expected, atol=1e-4)


def test_gradcam_stores_activations_and_gradients() -> None:
    """Los hooks guardan activaciones y gradientes con la forma de la capa."""

    model = GapNet()
    explainer = c.GradCAM(model, model.features)
    explainer.generate(torch.randn(1, 3, HEIGHT, WIDTH), 1)
    assert explainer.activations.shape == (1, 4, HEIGHT, WIDTH), "Activaciones mal."
    assert explainer.gradients.shape == (1, 4, HEIGHT, WIDTH), "Gradientes mal."
    # En una GAP el gradiente es constante espacialmente: w_ck / (h·w)
    expected = model.fc.weight[1].detach() / (HEIGHT * WIDTH)
    assert_close(explainer.gradients[0, :, 0, 0], expected)


@pytest.mark.parametrize("target_class", [None, 1])
def test_gradcam_dummy_cnn_upsampling(target_class: int | None) -> None:
    """Capa intermedia de DummyCNN: mapa del tamaño de la entrada y en [0, 1]."""

    model = DummyCNN(num_classes=NUM_CLASSES)
    explainer = c.GradCAM(model, model.features[2])
    out = explainer.generate(torch.randn(BATCH, 3, HEIGHT, WIDTH), target_class)
    assert out.shape == (BATCH, HEIGHT, WIDTH), "Se debe interpolar a la entrada."
    assert out.min() >= 0 and out.max() <= 1 + 1e-6, "Debe estar en [0, 1]."


def test_gradcam_lower_resolution_layer() -> None:
    """Si la capa tiene menor resolución, se interpola bilinealmente."""

    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, NUM_CLASSES),
    )
    explainer = c.GradCAM(model, model[1])
    out = explainer.generate(torch.randn(1, 3, HEIGHT, WIDTH), 0)
    assert explainer.activations.shape[-2:] == (HEIGHT // 2, WIDTH // 2)
    assert out.shape == (1, HEIGHT, WIDTH), "Forma incorrecta tras interpolar."


def test_gradcam_remove_hooks() -> None:
    """Tras ``remove_hooks`` los hooks dejan de actualizar las activaciones."""

    model = GapNet()
    explainer = c.GradCAM(model, model.features)
    assert len(explainer.handles) == 2, "Deben registrarse 2 hooks en el constructor."
    explainer.remove_hooks()
    model(torch.randn(1, 3, HEIGHT, WIDTH))
    assert explainer.activations is None, "El hook no debería ejecutarse ya."


def test_guided_gradcam_product() -> None:
    """Guided Grad-CAM = Grad-CAM (difundido a canales) ⊙ Guided Backprop."""

    gradcam_map = torch.tensor([[[0.0, 1.0], [0.5, 0.25]]])
    guided = torch.ones(1, 3, 2, 2) * torch.tensor([1.0, -2.0, 4.0]).view(1, 3, 1, 1)
    out = c.guided_gradcam(gradcam_map, guided)
    assert out.shape == (1, 3, 2, 2), "Forma incorrecta."
    assert_close(out[0, 1], torch.tensor([[0.0, -2.0], [-1.0, -0.5]]))


def test_attention_rollout_hand_value() -> None:
    """Atención uniforme de 2 tokens en 2 capas, calculado a mano."""

    attn = torch.full((1, 2, 2, 2), 0.5)  # [batch, heads, tokens, tokens]
    out = c.attention_rollout([attn, attn])
    expected = torch.tensor([[[0.625, 0.375], [0.375, 0.625]]])
    assert out.shape == (1, 2, 2), "Forma incorrecta."
    assert_close(out, expected)


def test_attention_rollout_identity_and_rows() -> None:
    """Atención identidad → rollout identidad; con atención aleatoria las filas suman 1."""

    eye = torch.eye(4).expand(BATCH, 3, 4, 4)
    assert_close(c.attention_rollout([eye, eye, eye]), torch.eye(4).expand(BATCH, 4, 4))
    attns = [torch.softmax(torch.randn(BATCH, 3, 4, 4), dim=-1) for _ in range(3)]
    out = c.attention_rollout(attns)
    assert_close(out.sum(dim=-1), torch.ones(BATCH, 4))
