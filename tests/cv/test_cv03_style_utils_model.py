"""
Tests de CV03 · Style transfer: matriz de Gram, normalización, AdaIN y hooks.
"""

import pytest
import torch
from helpers import DummyCNN, assert_close, load
from torch import nn

utils = load("cv/03_segmentation_style_transfer/style_transfer/utils.py")
model_mod = load("cv/03_segmentation_style_transfer/style_transfer/model.py")

FEATURES = torch.tensor([[[[1.0, 2.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]]])
EXPECTED_GRAM_RAW = torch.tensor([[[6.0, 1.0], [1.0, 2.0]]])
MEANS = [0.485, 0.456, 0.406]
STDS = [0.229, 0.224, 0.225]
DUMMY_INPUTS = torch.randn(1, 3, 12, 12)


@pytest.mark.parametrize(("mode", "divisor"), [("none", 1.0), ("hw", 4.0), ("chw", 8.0)])
def test_gram_matrix_hand_values(mode: str, divisor: float) -> None:
    """
    Gram calculada a mano con las tres normalizaciones.
    """

    gram = utils.gram_matrix(FEATURES, normalize=mode)
    assert gram.shape == (1, 2, 2), "Forma [batch, c, c]."
    assert_close(gram, EXPECTED_GRAM_RAW / divisor)


def test_gram_matrix_properties() -> None:
    """
    Simétrica, diagonal no negativa, por lote y error con normalización desconocida.
    """

    x = torch.randn(3, 5, 7, 6)
    gram = utils.gram_matrix(x)
    assert gram.shape == (3, 5, 5), "Debe haber una Gram por elemento del lote."
    assert_close(gram, gram.transpose(1, 2))
    assert torch.all(torch.diagonal(gram, dim1=1, dim2=2) >= 0), "Diagonal >= 0."
    expected = torch.einsum("bchw,bdhw->bcd", x, x) / (5 * 7 * 6)
    assert_close(gram, expected)
    with pytest.raises(ValueError):
        utils.gram_matrix(x, normalize="max")


def test_gram_matrix_ignores_spatial_order() -> None:
    """
    La Gram no depende de dónde están las cosas: permutar píxeles no la cambia.
    """

    x = torch.randn(1, 4, 5, 5)
    perm = torch.randperm(25)
    x_perm = x.flatten(2)[..., perm].reshape(1, 4, 5, 5)
    assert_close(utils.gram_matrix(x), utils.gram_matrix(x_perm))


def test_normalize_denormalize() -> None:
    """
    Normaliza canal a canal y ``denormalize`` lo deshace (y recorta a [0, 1]).
    """

    x = torch.rand(2, 3, 4, 4)
    norm = utils.normalize(x, MEANS, STDS)
    assert_close(norm[:, 1], (x[:, 1] - 0.456) / 0.224)
    assert_close(utils.denormalize(norm, MEANS, STDS), x)
    big = utils.denormalize(torch.full((1, 3, 2, 2), 100.0), MEANS, STDS)
    assert torch.all(big == 1.0), "denormalize debe recortar a [0, 1]."


def test_adain() -> None:
    """
    La salida tiene la media y desviación (por canal) del estilo.
    """

    content = torch.randn(2, 4, 8, 8) * 3 + 1
    style = torch.randn(2, 4, 6, 6) * 0.5 - 2
    out = utils.adain(content, style, eps=0.0)
    assert out.shape == content.shape, "Forma incorrecta."
    assert_close(out.mean(dim=(2, 3)), style.mean(dim=(2, 3)), atol=1e-4)
    assert_close(
        out.std(dim=(2, 3), unbiased=False), style.std(dim=(2, 3), unbiased=False), atol=1e-4
    )


def _n_layers(net: nn.Module, kind: type[nn.Module]) -> int:
    return sum(isinstance(x, kind) for x in net.modules())


def test_register_and_remove_hooks() -> None:
    """
    Un hook por Conv2d, una activación por Conv2d y limpieza completa.
    """

    cnn = model_mod.CustomModel(DummyCNN())
    assert len(cnn.hooks) == 0, "No debe haber hooks al crear el objeto."
    cnn._register_hooks()
    n_convs = _n_layers(cnn.model, nn.Conv2d)
    assert len(cnn.hooks) == n_convs, "Debe haber un hook por Conv2d."
    cnn.model(DUMMY_INPUTS)
    assert len(cnn.activations) == n_convs, "Una activación por Conv2d."
    assert torch.allclose(cnn.activations[0], cnn.model.features[0](DUMMY_INPUTS)), (
        "La primera activación es la salida de la primera Conv2d."
    )
    cnn._remove_hooks_and_activations()
    assert len(cnn.hooks) == 0 and len(cnn.activations) == 0, "Limpieza incompleta."
    cnn.model(DUMMY_INPUTS)
    assert len(cnn.activations) == 0, "Los hooks deben estar eliminados."


def test_hooks_other_layer_type() -> None:
    """
    Con ``layer_type=nn.ReLU`` se enganchan las ReLU (salidas no negativas).
    """

    net = model_mod.build_tiny_vgg()
    cnn = model_mod.CustomModel(net, layer_type=nn.ReLU)
    acts = cnn.obtain_activations(DUMMY_INPUTS, [0, 3], use_grad=False)
    assert all(torch.all(a >= 0) for a in acts.values()), "Son salidas de ReLU."
    assert acts[3].shape == (1, 16, 6, 6), "La 4.ª ReLU está tras el MaxPool."


@pytest.mark.parametrize("layers", [[0, 2], [1], [0, 1, 2, 3]])
@pytest.mark.parametrize("use_grad", [True, False])
def test_obtain_activations(layers: list[int], use_grad: bool) -> None:
    """
    Devuelve un dict {capa: activación}, respeta use_grad y no deja hooks.
    """

    cnn = model_mod.CustomModel(model_mod.build_tiny_vgg())
    x = DUMMY_INPUTS.clone().requires_grad_(use_grad)
    acts = cnn.obtain_activations(x, layers, use_grad=use_grad)
    assert isinstance(acts, dict) and set(acts) == set(layers), "Claves incorrectas."
    for act in acts.values():
        assert isinstance(act, torch.Tensor) and act.shape[0] == 1, "Activación incorrecta."
        assert act.requires_grad is use_grad, "requires_grad debe seguir a use_grad."
    assert len(cnn.hooks) == 0 and len(cnn.activations) == 0, (
        "Hay que eliminar hooks y activaciones al terminar."
    )


def test_obtain_activations_values_and_errors() -> None:
    """
    Los valores coinciden con un forward manual y un índice inválido da ValueError.
    """

    net = model_mod.build_tiny_vgg()
    cnn = model_mod.CustomModel(net)
    acts = cnn.obtain_activations(DUMMY_INPUTS, [1], use_grad=False)
    expected = net[:3](DUMMY_INPUTS)  # conv, relu, conv
    assert_close(acts[1], expected)
    with pytest.raises(ValueError):
        cnn.obtain_activations(DUMMY_INPUTS, [0, 10], use_grad=False)
    assert len(cnn.hooks) == 0, "Incluso tras el error no deben quedar hooks."
