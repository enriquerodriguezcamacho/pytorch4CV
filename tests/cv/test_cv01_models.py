"""
Tests de CV01 · utilidades de transfer learning sobre modelos.
"""

import pytest
import torch
from helpers import DummyCNN, load
from torch import nn
from torchvision import models

m = load("cv/01_introduction_transfer_learning/models.py")

NUM_CLASSES = 5


class CustomNet(nn.Module):
    """Clase propia sin ``fc`` ni ``classifier``: la cabeza se llama ``head``."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.neck = nn.Linear(4, 8)
        self.head = nn.Linear(8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.neck(self.body(x).flatten(1))))


def tiny_vgg() -> nn.Module:
    """
    VGG en miniatura con la misma estructura que ``torchvision.models.VGG``
    (``features`` + ``avgpool`` + ``classifier`` secuencial). Se usa la clase
    oficial ``VGG`` pero se reemplaza el clasificador para que sea rápida.
    """

    features = models.vgg.make_layers([8, "M", 16, "M"])
    model = models.VGG.__new__(models.VGG)
    nn.Module.__init__(model)
    model.features = features
    model.avgpool = nn.AdaptiveAvgPool2d((2, 2))
    model.classifier = nn.Sequential(
        nn.Linear(16 * 2 * 2, 16), nn.ReLU(), nn.Dropout(), nn.Linear(16, 1000)
    )
    return model


def test_get_final_layer_resnet() -> None:
    """En ResNet la cabeza es ``fc``."""

    model = models.resnet18(weights=None)
    name, layer = m.get_final_layer(model)
    assert name == "fc", "En ResNet la última capa es 'fc'."
    assert layer is model.fc, "Debe devolverse el propio módulo, no una copia."


def test_get_final_layer_alexnet() -> None:
    """En AlexNet la cabeza es ``classifier.6``."""

    model = models.alexnet(weights=None)
    name, layer = m.get_final_layer(model)
    assert name == "classifier.6", "En AlexNet la última capa es classifier[6]."
    assert layer is model.classifier[6], "Capa incorrecta."


def test_get_final_layer_custom_and_error() -> None:
    """Clase propia: última ``nn.Linear``; sin lineales: ``ValueError``."""

    name, _ = m.get_final_layer(CustomNet())
    assert name == "head", "Debe encontrarse la última nn.Linear de la clase propia."
    with pytest.raises(ValueError):
        m.get_final_layer(nn.Sequential(nn.Conv2d(3, 3, 3)))


@pytest.mark.parametrize(
    "builder, attr",
    [
        (lambda: models.resnet18(weights=None), lambda mod: mod.fc),
        (tiny_vgg, lambda mod: mod.classifier[-1]),
        (lambda: models.alexnet(weights=None), lambda mod: mod.classifier[-1]),
        (CustomNet, lambda mod: mod.head),
        (lambda: DummyCNN(3), lambda mod: mod.fc),
    ],
)
def test_replace_last_layer(builder, attr) -> None:
    """La nueva cabeza tiene ``num_classes`` salidas y el forward funciona."""

    model = builder()
    in_features = attr(model).in_features
    out = m.replace_last_layer(model, NUM_CLASSES)
    assert out is model, "El cambio debe hacerse in place y devolver el modelo."
    head = attr(model)
    assert isinstance(head, nn.Linear), "La cabeza debe ser nn.Linear."
    assert head.out_features == NUM_CLASSES, "Número de salidas incorrecto."
    assert head.in_features == in_features, "in_features debe conservarse."
    size = 64 if isinstance(model, models.AlexNet) else 32
    with torch.no_grad():
        logits = model.eval()(torch.randn(2, 3, size, size))
    assert logits.shape == (2, NUM_CLASSES), "Forma de la salida incorrecta."


def test_freeze_backbone_resnet() -> None:
    """Solo ``fc`` queda entrenable."""

    model = m.freeze_backbone(models.resnet18(weights=None))
    for name, p in model.named_parameters():
        expected = name.startswith("fc.")
        assert p.requires_grad == expected, f"requires_grad incorrecto en {name}."
    trainable = m.get_trainable_parameters(model)
    assert len(trainable) == 2, "Solo deben quedar weight y bias de fc."
    assert m.count_parameters(model, trainable_only=True) == 512 * 1000 + 1000, (
        "Número de parámetros entrenables incorrecto."
    )


def test_count_parameters_resnet18() -> None:
    """ResNet-18 tiene 11 689 512 parámetros."""

    assert m.count_parameters(models.resnet18(weights=None)) == 11_689_512, (
        "Recuento total incorrecto."
    )


def test_unfreeze_layers() -> None:
    """Descongelado progresivo: ``layer4`` + ``fc`` entrenables, ``layer3`` no."""

    model = m.freeze_backbone(models.resnet18(weights=None))
    m.unfreeze_layers(model, ["layer4"])
    assert all(p.requires_grad for p in model.layer4.parameters()), "layer4 congelada."
    assert all(p.requires_grad for p in model.fc.parameters()), "fc debe seguir activa."
    assert not any(p.requires_grad for p in model.layer3.parameters()), (
        "layer3 no debe descongelarse."
    )


def test_unfreeze_layers_exact_prefix() -> None:
    """``"layer1"`` no debe descongelar una capa llamada ``"layer10"``."""

    model = nn.Module()
    model.layer1 = nn.Linear(2, 2)
    model.layer10 = nn.Linear(2, 2)
    for p in model.parameters():
        p.requires_grad = False
    m.unfreeze_layers(model, ["layer1"])
    assert model.layer1.weight.requires_grad, "layer1 debería estar activa."
    assert not model.layer10.weight.requires_grad, "Prefijo aplicado sin el punto."


def test_discriminative_lr_param_groups() -> None:
    """Learning rates decrecientes hacia la entrada y grupo extra para el resto."""

    model = models.resnet18(weights=None)
    groups = m.discriminative_lr_param_groups(
        model, ["layer3", "layer4", "fc"], base_lr=1e-2, decay=0.1
    )
    lrs = [g["lr"] for g in groups]
    assert lrs == pytest.approx([1e-5, 1e-4, 1e-3, 1e-2]), "Learning rates incorrectos."
    n_params = sum(len(g["params"]) for g in groups)
    assert n_params == len(list(model.parameters())), "Faltan o sobran parámetros."
    assert groups[-1]["params"][0] is model.fc.weight, "El último grupo debe ser fc."
    optimizer = torch.optim.SGD(groups, lr=0.1)
    assert len(optimizer.param_groups) == 4, "Los grupos deben valer para SGD."


def test_discriminative_lr_skips_frozen() -> None:
    """Los parámetros congelados no se incluyen y no hay grupos vacíos."""

    model = m.freeze_backbone(models.resnet18(weights=None))
    groups = m.discriminative_lr_param_groups(model, ["layer4", "fc"], base_lr=1.0)
    assert len(groups) == 1, "Solo fc es entrenable: un único grupo."
    assert groups[0]["lr"] == pytest.approx(1.0), "fc debe tener base_lr."
