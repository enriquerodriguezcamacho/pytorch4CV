"""
Tests de CV01 · estrategias de transfer learning (estilo práctica 1).
"""

from unittest.mock import patch

import pytest
import torch
from helpers import DummyCNN, load
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

s = load("cv/01_introduction_transfer_learning/strategies.py")
cfg = load("cv/01_introduction_transfer_learning/config.py")

NUM_CLASSES = 3
NUM_SAMPLES = 8
BATCH = 4
IMG = 32


def make_loader(shuffle: bool = False) -> DataLoader:
    """Mini dataset aleatorio de imágenes 32×32."""

    g = torch.Generator().manual_seed(0)
    x = torch.randn(NUM_SAMPLES, 3, IMG, IMG, generator=g)
    y = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=shuffle)


def make_resnet() -> nn.Module:
    """ResNet-18 sin pesos con cabeza de 3 clases."""

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, NUM_CLASSES)
    return model


def make_config(epochs: int = 1) -> object:
    return cfg.TrainConfig(epochs=epochs, lr=1e-2, feature_batch_size=4)


def test_strategy_is_abstract() -> None:
    """``TransferStrategy`` no se puede instanciar."""

    with pytest.raises(TypeError):
        s.TransferStrategy(make_config())


def test_train_one_epoch_updates_parameters() -> None:
    """Un paso de entrenamiento cambia los parámetros y deja el modelo en train."""

    model = DummyCNN(NUM_CLASSES)
    strategy = s.FullFineTuningStrategy(make_config())
    old = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.eval()
    loss = strategy._train_one_epoch(model, make_loader(), optimizer)
    assert isinstance(loss, float) and loss > 0, "Debe devolver la pérdida media."
    assert model.training, "El modelo debe quedar en modo train."
    changed = [torch.norm(n - o) > 0 for n, o in zip(model.parameters(), old)]
    assert all(changed), "Todos los parámetros deberían haberse actualizado."


def test_validate_metrics_and_eval_mode() -> None:
    """``_validate`` usa modo eval, no acumula gradientes y calcula bien la accuracy."""

    model = DummyCNN(NUM_CLASSES)
    strategy = s.LastLayerStrategy(make_config())
    loader = make_loader()
    metrics = strategy._validate(model, loader)
    assert not model.training, "El modelo debe estar en modo eval."
    assert all(p.grad is None for p in model.parameters()), "No debe haber gradientes."
    with torch.no_grad():
        x, y = loader.dataset.tensors
        logits = model(x)
        acc = (logits.argmax(1) == y).float().mean().item()
        loss = nn.functional.cross_entropy(logits, y).item()
    assert metrics["accuracy"] == pytest.approx(acc), "Accuracy incorrecta."
    assert metrics["loss"] == pytest.approx(loss, rel=1e-4), "Pérdida incorrecta."


def test_last_layer_strategy_only_trains_head() -> None:
    """Solo cambia la cabeza; el optimizador solo tiene sus parámetros."""

    model = make_resnet()
    strategy = s.LastLayerStrategy(make_config())
    model, optimizer = strategy._prepare(model)
    params = [p for g in optimizer.param_groups for p in g["params"]]
    assert len(params) == 2, "El optimizador solo debe contener weight y bias de fc."
    old_conv = model.conv1.weight.detach().clone()
    old_fc = model.fc.weight.detach().clone()
    strategy._train_one_epoch(model, make_loader(), optimizer)
    assert torch.equal(model.conv1.weight, old_conv), "El backbone no debe cambiar."
    assert torch.norm(model.fc.weight - old_fc) > 0, "La cabeza debe actualizarse."


def test_full_fine_tuning_param_groups() -> None:
    """Todo entrenable y lr del backbone = lr * backbone_lr_factor."""

    model = make_resnet()
    for p in model.parameters():
        p.requires_grad = False
    config = make_config()
    model, optimizer = s.FullFineTuningStrategy(config)._prepare(model)
    assert all(p.requires_grad for p in model.parameters()), "Todo debe entrenarse."
    backbone_group, head_group = optimizer.param_groups
    assert backbone_group["lr"] == pytest.approx(config.lr * config.backbone_lr_factor)
    assert head_group["lr"] == pytest.approx(config.lr), "lr de la cabeza incorrecto."
    total = len(backbone_group["params"]) + len(head_group["params"])
    assert total == len(list(model.parameters())), "Faltan parámetros."


@pytest.mark.parametrize("epochs", [1, 3])
def test_execute_calls_train_and_validate(epochs: int) -> None:
    """``execute`` llama ``epochs`` veces a entrenar y validar (con mock)."""

    strategy = s.LastLayerStrategy(make_config(epochs))
    with (
        patch.object(strategy, "_train_one_epoch", return_value=0.5) as train,
        patch.object(
            strategy, "_validate", return_value={"loss": 0.4, "accuracy": 0.9}
        ) as val,
    ):
        history = strategy.execute(DummyCNN(NUM_CLASSES), make_loader(), make_loader())
    assert train.call_count == epochs, "Número de llamadas a _train_one_epoch."
    assert val.call_count == epochs, "Número de llamadas a _validate."
    assert history["train_loss"] == [0.5] * epochs, "Historial de train incorrecto."
    assert history["val_accuracy"] == [0.9] * epochs, "Historial de val incorrecto."


def test_full_fine_tuning_execute_resnet() -> None:
    """Ejecución real de full fine-tuning: cambian backbone y cabeza."""

    model = make_resnet()
    old_conv = model.conv1.weight.detach().clone()
    history = s.FullFineTuningStrategy(make_config()).execute(
        model, make_loader(), make_loader()
    )
    assert len(history["val_loss"]) == 1, "Una entrada por época."
    assert 0.0 <= history["val_accuracy"][0] <= 1.0, "Accuracy fuera de rango."
    assert torch.norm(model.conv1.weight - old_conv) > 0, "El backbone debe cambiar."


def test_extract_features_shapes() -> None:
    """Las features de ResNet-18 tienen 512 dimensiones y no llevan grafo."""

    strategy = s.FeatureExtractionStrategy(make_config())
    model = make_resnet()
    extractor = strategy._build_feature_extractor(model)
    feats, labels = strategy._extract_features(extractor, make_loader())
    assert feats.shape == (NUM_SAMPLES, 512), "Forma de las features incorrecta."
    assert labels.shape == (NUM_SAMPLES,), "Forma de las etiquetas incorrecta."
    assert not feats.requires_grad, "Las features deben calcularse con no_grad."
    assert isinstance(model.fc, nn.Linear), "El modelo original no debe modificarse."


def test_feature_extraction_execute() -> None:
    """Solo cambia la cabeza del modelo original y el modelo funciona end-to-end."""

    model = make_resnet()
    old_conv = model.conv1.weight.detach().clone()
    old_fc = model.fc.weight.detach().clone()
    strategy = s.FeatureExtractionStrategy(make_config(epochs=2))
    history = strategy.execute(model, make_loader(), make_loader())
    assert len(history["train_loss"]) == 2, "Una entrada por época."
    assert torch.equal(model.conv1.weight, old_conv), "El backbone no debe cambiar."
    assert torch.norm(model.fc.weight - old_fc) > 0, "La cabeza debe entrenarse."
    with torch.no_grad():
        out = model.eval()(torch.randn(2, 3, IMG, IMG))
    assert out.shape == (2, NUM_CLASSES), "El modelo debe seguir funcionando."
