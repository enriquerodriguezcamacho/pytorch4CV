"""
Tests de los ejercicios del tema 1 de los apuntes y del ``DataManager`` de la práctica 1.
"""

import pytest
import torch
from helpers import assert_close, load
from PIL import Image
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision.models import resnet18

a = load("cv/01_foundations_cnn/apuntes.py")
d = load("cv/01_foundations_cnn/data_manager.py")


def test_conv_output_shape_exercise_values() -> None:
    """(3, 224, 224), 64 filtros 7×7, s=2, p=3 → (64, 112, 112)."""

    assert a.conv_output_shape((3, 224, 224), 64, 7, 2, 3) == (64, 112, 112)


@pytest.mark.parametrize("k,s,p,dil", [(3, 1, 0, 1), (5, 2, 2, 1), (3, 2, 1, 2)])
def test_conv_output_shape_matches_conv2d(k: int, s: int, p: int, dil: int) -> None:
    """Coincide con ``nn.Conv2d``."""

    conv = nn.Conv2d(3, 8, k, stride=s, padding=p, dilation=dil)
    expected = tuple(conv(torch.zeros(1, 3, 33, 40)).shape[1:])
    assert a.conv_output_shape((3, 33, 40), 8, k, s, p, dil) == expected


def test_receptive_field_five_layers() -> None:
    """Cinco capas 3×3: 1, 3, 5, 7, 9, 11."""

    assert a.receptive_field_per_layer([3] * 5) == [1, 3, 5, 7, 9, 11]


def test_adapt_resnet_to_grayscale() -> None:
    """Acepta 1 canal, promedia los pesos RGB y sustituye la cabeza."""

    model = resnet18(weights=None)
    expected = model.conv1.weight.data.mean(dim=1, keepdim=True).clone()
    adapted = a.adapt_resnet_to_grayscale(model, 10).eval()
    assert_close(adapted.conv1.weight.data, expected)
    assert adapted.conv1.stride == (2, 2)
    assert adapted(torch.zeros(2, 1, 64, 64)).shape == (2, 10)


def test_build_train_transforms() -> None:
    """Devuelve un tensor del tamaño pedido y es aleatoria según la semilla."""

    image = Image.fromarray((torch.rand(40, 40, 3) * 255).byte().numpy())
    transform = a.build_train_transforms(32)
    assert transform(image).shape == (3, 32, 32)
    assert not torch.equal(transform(image), transform(image))


def test_prepare_vgg_feature_extractor() -> None:
    """Congela solo ``features`` y cambia la última capa."""

    model = nn.Module()
    model.features = nn.Sequential(nn.Conv2d(3, 4, 3))
    model.classifier = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 7))
    a.prepare_vgg_feature_extractor(model, 10)
    assert all(not p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    assert model.classifier[-1].out_features == 10


def test_data_manager_validates_config() -> None:
    """Rechaza una configuración sin nombre o con un *dataset* desconocido."""

    with pytest.raises(ValueError):
        d.DataManager({"data": {}})
    with pytest.raises(ValueError):
        d.DataManager({"data": {"name": "NO_EXISTE"}})


def test_data_manager_loaders() -> None:
    """Barajado en train, sin barajar en val, con imágenes 224×224 normalizadas."""

    manager = d.DataManager({"data": {"name": "FAKE", "batch_size": 2}})
    assert manager.num_classes == 3
    with pytest.raises(ValueError):
        manager.get_train_loader()
    manager.load_data()
    assert manager.class_names == ["a", "b", "c"]
    train, val = manager.get_train_loader(), manager.get_val_loader()
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(val.sampler, SequentialSampler)
    images, labels = next(iter(train))
    assert images.shape == (2, 3, 224, 224) and labels.shape == (2,)
