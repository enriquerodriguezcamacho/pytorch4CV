"""
Tests de CV03 · Style transfer: pérdida de Gatys y Trainer que optimiza la imagen.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from helpers import DummyCNN, load
from PIL import Image

cfg_mod = load("cv/03_segmentation_style_transfer/style_transfer/config.py")
loss_mod = load("cv/03_segmentation_style_transfer/style_transfer/loss.py")
model_mod = load("cv/03_segmentation_style_transfer/style_transfer/model.py")
train_mod = load("cv/03_segmentation_style_transfer/style_transfer/train.py")

# Features (calculadas a mano)
CONTENT_FEATURES = {
    2: torch.tensor([[[[1.0, 2.0], [0.0, 0.0]]]]),
    4: torch.zeros(1, 2, 1, 2),
}
STYLE_FEATURES = {
    0: torch.tensor([[[0.5]]]),
    1: torch.tensor([[[0.5, 0.0], [0.0, 1.0]]]),
}
SYNTHESIZED_FEATURES = {
    0: torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
    1: torch.tensor([[[[1.0, 1.0]], [[0.0, 2.0]]]]),
    2: torch.tensor([[[[0.0, 2.0], [1.0, 1.0]]]]),
    4: torch.ones(1, 2, 1, 2),
}

ALPHA = 0.2
EXPECTED_CONTENT_LOSS = torch.tensor(0.175)  # 0.2 * (0.75 + 1) / 2
BETA = 0.8
EXPECTED_STYLE_LOSS = torch.tensor(0.075)  # 0.8 * (0.0625 + 0.125) / 2
STYLE_WEIGHTS = {0: 0.25, 1: 0.75}
EXPECTED_WEIGHTED_STYLE_LOSS = torch.tensor(0.0875)  # 0.8 * (0.25·0.0625 + 0.75·0.125)
GAMMA = 0.3
SYNTHESIZED_IMAGE = torch.stack(
    [torch.tensor([[0.0, 2.0], [3.0, 1.0]]), torch.zeros(2, 2), torch.ones(2, 2)]
).unsqueeze(0)
EXPECTED_TV_LOSS = torch.tensor(0.1)  # 0.3 * 8 / 24
EXPECTED_TV_SQUARED_LOSS = torch.tensor(0.225)  # 0.3 * 18 / 24

LOSS = loss_mod.StyleTransferLoss(
    CONTENT_FEATURES, STYLE_FEATURES, model_mod.CustomModel(DummyCNN())
)
DIVIDE_BY = 2
SIZE = 16


def test_content_loss() -> None:
    """
    Valor a mano y escalado lineal con alpha.
    """

    loss_1 = LOSS._content_loss(SYNTHESIZED_FEATURES, alpha=ALPHA)
    loss_2 = LOSS._content_loss(SYNTHESIZED_FEATURES, alpha=ALPHA / DIVIDE_BY)
    assert torch.isclose(loss_1, EXPECTED_CONTENT_LOSS), "Valor incorrecto."
    assert torch.isclose(loss_2, EXPECTED_CONTENT_LOSS / DIVIDE_BY), (
        f"La pérdida debe dividirse entre {DIVIDE_BY}."
    )


def test_style_loss() -> None:
    """
    Valor a mano (pesos uniformes), escalado con beta y pesos por capa.
    """

    loss_1 = LOSS._style_loss(SYNTHESIZED_FEATURES, beta=BETA)
    loss_2 = LOSS._style_loss(SYNTHESIZED_FEATURES, beta=BETA / DIVIDE_BY)
    assert torch.isclose(loss_1, EXPECTED_STYLE_LOSS), "Valor incorrecto."
    assert torch.isclose(loss_2, EXPECTED_STYLE_LOSS / DIVIDE_BY), (
        f"La pérdida debe dividirse entre {DIVIDE_BY}."
    )
    weighted = loss_mod.StyleTransferLoss(
        CONTENT_FEATURES,
        STYLE_FEATURES,
        model_mod.CustomModel(DummyCNN()),
        style_layer_weights=STYLE_WEIGHTS,
    )
    out = weighted._style_loss(SYNTHESIZED_FEATURES, beta=BETA)
    assert torch.isclose(out, EXPECTED_WEIGHTED_STYLE_LOSS), "Pesos por capa ignorados."


@pytest.mark.parametrize(
    ("squared", "expected"), [(False, EXPECTED_TV_LOSS), (True, EXPECTED_TV_SQUARED_LOSS)]
)
def test_total_variation_loss(squared: bool, expected: torch.Tensor) -> None:
    """
    TV con valor absoluto y con cuadrado; escalado lineal con gamma.
    """

    loss = loss_mod.StyleTransferLoss(
        CONTENT_FEATURES, STYLE_FEATURES, LOSS.model, tv_squared=squared
    )
    tv_1 = loss._total_variation_loss(SYNTHESIZED_IMAGE, gamma=GAMMA)
    tv_2 = loss._total_variation_loss(SYNTHESIZED_IMAGE, gamma=GAMMA / DIVIDE_BY)
    assert torch.isclose(tv_1, expected), "Valor incorrecto."
    assert torch.isclose(tv_2, expected / DIVIDE_BY), "Debe escalar con gamma."
    flat = torch.full((1, 3, 5, 5), 0.7)
    assert loss._total_variation_loss(flat, gamma=1.0) == 0, "Imagen constante: TV 0."


def _tiny_features(x: torch.Tensor, layers: list[int], gram: bool) -> dict:
    cnn = model_mod.CustomModel(model_mod.build_tiny_vgg())
    return cnn, cnn.obtain_activations(x, layers, use_grad=False)


def test_forward() -> None:
    """
    El forward devuelve las cuatro pérdidas, total = suma, y el gradiente llega a la
    imagen.
    """

    cnn = model_mod.CustomModel(model_mod.build_tiny_vgg())
    content, style = torch.randn(1, 3, SIZE, SIZE), torch.randn(1, 3, SIZE, SIZE)
    content_feats = cnn.obtain_activations(content, [3], use_grad=False)
    style_feats = {
        k: model_mod.CustomModel(cnn.model).obtain_activations(style, [k], False)[k]
        for k in [0, 1]
    }
    style_grams = {k: v.flatten(2) @ v.flatten(2).transpose(1, 2) for k, v in style_feats.items()}
    style_grams = {k: g / style_feats[k][0].numel() for k, g in style_grams.items()}
    criterion = loss_mod.StyleTransferLoss(content_feats, style_grams, cnn)

    x = content.clone().requires_grad_(True)
    c_loss, s_loss, tv_loss, loss = criterion(x)
    assert torch.isclose(loss, c_loss + s_loss + tv_loss), "Total = suma de términos."
    assert torch.isclose(c_loss, torch.tensor(0.0), atol=1e-6), (
        "Si la imagen es la de contenido, la content loss es 0."
    )
    assert s_loss > 0 and tv_loss > 0, "Estilo y TV deberían ser positivas."
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, "El gradiente debe llegar a x."


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _make_cfg(optimizer: type = torch.optim.Adam, lr: float = 5e-2):
    return cfg_mod.Config(
        model=cfg_mod.ModelConfig(size_images=SIZE),
        training=cfg_mod.TrainingConfig(epochs=3, lr=lr, optimizer=optimizer),
    )


@pytest.fixture
def trainer():
    """
    Trainer con imágenes aleatorias y la VGG diminuta.
    """

    content = torch.randn(1, 3, SIZE, SIZE)
    style = torch.randn(1, 3, SIZE, SIZE)
    return train_mod.Trainer(content, style, cfg=_make_cfg())


def test_init_content_features(trainer) -> None:
    """
    Dict {capa: features} con las capas de contenido de la configuración.
    """

    feats = trainer.content_features
    assert isinstance(feats, dict), "Tipo incorrecto."
    assert list(feats) == trainer.cfg.loss.layers_content, "Claves incorrectas."
    for act in feats.values():
        assert act.shape[0] == 1 and act.dim() == 4, "Forma incorrecta."
        assert not act.requires_grad, "Las features de contenido son constantes."


def test_init_style_features(trainer) -> None:
    """
    Dict {capa: Gram} simétricas y constantes.
    """

    feats = trainer.style_features
    assert sorted(feats) == sorted(trainer.cfg.loss.layers_style), "Claves incorrectas."
    for gram in feats.values():
        assert gram.dim() == 3 and gram.shape[1] == gram.shape[2], "Debe ser una Gram."
        assert torch.allclose(gram, gram.transpose(1, 2)), "La Gram es simétrica."
        assert torch.all(torch.diagonal(gram, dim1=1, dim2=2) >= 0), "Diagonal >= 0."
        assert not gram.requires_grad, "Las Gram de estilo son constantes."


def test_init_synthesized_image_random(trainer) -> None:
    """
    Ruido N(0, 1) de la forma pedida, hoja y con gradientes.
    """

    img = trainer._init_synthesized_image(random=True, size_images=64)
    assert img.shape == (1, 3, 64, 64), "Forma incorrecta."
    assert torch.isclose(img.mean(), torch.tensor(0.0), atol=5e-2), "Media ~ 0."
    assert torch.isclose(img.std(), torch.tensor(1.0), atol=5e-2), "Desviación ~ 1."
    assert img.requires_grad and img.is_leaf, "Debe ser un tensor hoja con gradientes."


def test_init_synthesized_image_as_content() -> None:
    """
    Copia independiente de la imagen de contenido; sin contenido da ValueError.
    """

    content = torch.randn(1, 3, 10, 12)
    img = train_mod.Trainer._init_synthesized_image(False, 999, content)
    assert img.shape == content.shape and torch.equal(img, content), "Debe ser una copia."
    assert img.requires_grad and img.is_leaf, "Debe ser un tensor hoja con gradientes."
    with torch.no_grad():
        img.add_(1.0)
    assert not torch.equal(img, content), "La copia no debe compartir memoria."
    with pytest.raises(ValueError):
        train_mod.Trainer._init_synthesized_image(False, 8, None)


@pytest.mark.parametrize("optimizer", [torch.optim.Adam, torch.optim.LBFGS])
def test_train_one_epoch(optimizer: type) -> None:
    """
    Actualiza la imagen (Adam y LBFGS), guarda una vez las pérdidas y no toca la red.
    """

    content, style = torch.randn(1, 3, SIZE, SIZE), torch.randn(1, 3, SIZE, SIZE)
    lr = 1.0 if optimizer is torch.optim.LBFGS else 5e-2
    tr = train_mod.Trainer(content, style, cfg=_make_cfg(optimizer, lr))
    old_img = tr.synthesized_image.detach().clone()
    old_params = [p.clone() for p in tr.model.parameters()]
    tr._train_one_epoch()
    for name, values in tr.losses.items():
        assert len(values) == 1 and isinstance(values[0], float), f"Pérdida {name} mal."
    assert torch.mean(torch.abs(tr.synthesized_image.detach() - old_img)) > 0, (
        "La imagen no se ha actualizado."
    )
    for old, new in zip(old_params, tr.model.parameters()):
        assert torch.equal(old, new) and not new.requires_grad, "La red debe estar congelada."


def test_fit_calls_train_one_epoch(trainer) -> None:
    """
    ``fit`` llama a ``_train_one_epoch`` una vez por época.
    """

    def fake() -> None:
        for values in trainer.losses.values():
            values.append(0.5)

    epochs = 4
    with patch.object(trainer, "_train_one_epoch", side_effect=fake) as mock:
        out = trainer.fit(epochs=epochs)
    assert mock.call_count == epochs, "Una llamada por época."
    assert out.shape == (1, 3, SIZE, SIZE) and not out.requires_grad, "Salida incorrecta."


def test_fit_saves_images_and_reduces_loss(trainer, tmp_path: Path) -> None:
    """
    Guarda las imágenes de las épocas 1, múltiplos de save_every y última; la pérdida
    baja.
    """

    save_dir = tmp_path / "synth"
    trainer.fit(epochs=6, save_every=4, save_dir=str(save_dir))
    saved = sorted(p.name for p in save_dir.iterdir())
    assert saved == ["1.png", "4.png", "6.png"], f"Imágenes guardadas: {saved}."
    assert len(trainer.losses["loss"]) == 6, "Una pérdida por época."
    assert trainer.losses["loss"][-1] < trainer.losses["loss"][0], "La pérdida debe bajar."


def test_from_paths(tmp_path: Path) -> None:
    """
    El constructor alternativo lee imágenes de disco (funciones dadas).
    """

    for name in ["content.png", "style.png"]:
        array = np.random.randint(0, 256, (20, 24, 3), dtype=np.uint8)
        Image.fromarray(array).save(tmp_path / name)
    tr = train_mod.Trainer.from_paths(
        str(tmp_path / "content.png"), str(tmp_path / "style.png"), cfg=_make_cfg()
    )
    assert tr.synthesized_image.shape == (1, 3, SIZE, SIZE), "Forma incorrecta."
