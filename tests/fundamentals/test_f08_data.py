"""
Tests de F08 · Pipeline de datos.
"""

import pytest
import torch
from helpers import assert_close, load
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import tv_tensors
from torchvision.transforms import v2

m = load("fundamentals/08_data_pipeline/data.py")

NUM_SAMPLES = 10
IMAGE_SIZE = 8
NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# TensorPairDataset
# ---------------------------------------------------------------------------
def test_tensor_pair_dataset_len_and_getitem() -> None:
    """``__len__`` y ``__getitem__`` devuelven lo esperado."""

    x = torch.randn(NUM_SAMPLES, 3, 4, 4)
    y = torch.arange(NUM_SAMPLES)
    ds = m.TensorPairDataset(x, y)
    assert len(ds) == NUM_SAMPLES, "len(dataset) incorrecto."
    xi, yi = ds[3]
    assert torch.equal(xi, x[3]), "La entrada de la muestra 3 no coincide."
    assert yi.item() == 3, "La etiqueta de la muestra 3 no coincide."


def test_tensor_pair_dataset_transform() -> None:
    """La transformación se aplica solo a la entrada."""

    x, y = torch.ones(4, 2), torch.zeros(4)
    ds = m.TensorPairDataset(x, y, transform=lambda t: t * 5)
    xi, yi = ds[0]
    assert torch.equal(xi, torch.full((2,), 5.0)), "No se aplicó la transformación."
    assert yi.item() == 0, "La etiqueta no debe transformarse."


def test_tensor_pair_dataset_raises() -> None:
    """Distinto número de muestras → ``ValueError``."""

    with pytest.raises(ValueError):
        m.TensorPairDataset(torch.zeros(3, 2), torch.zeros(4))


def test_tensor_pair_dataset_with_dataloader() -> None:
    """Funciona con ``DataLoader`` (default_collate apila las muestras)."""

    ds = m.TensorPairDataset(torch.randn(NUM_SAMPLES, 5), torch.arange(NUM_SAMPLES))
    batches = list(DataLoader(ds, batch_size=4, drop_last=True))
    assert len(batches) == NUM_SAMPLES // 4, "drop_last debería descartar 1 lote."
    assert batches[0][0].shape == (4, 5), "Forma del lote incorrecta."


# ---------------------------------------------------------------------------
# SyntheticSegmentationDataset
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hflip, vflip", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
def test_segmentation_deterministic_flips(hflip: float, vflip: float) -> None:
    """Con probabilidad 1 el flip se aplica a imagen y máscara."""

    ds = m.SyntheticSegmentationDataset(
        num_samples=2, image_size=IMAGE_SIZE, hflip_prob=hflip, vflip_prob=vflip
    )
    image, mask = ds[0]
    dims = [d for d, p in ((-1, hflip), (-2, vflip)) if p == 1.0]
    assert torch.equal(image, ds.images[0].flip(dims)), "Flip de la imagen incorrecto."
    assert torch.equal(mask, ds.masks[0].flip(dims)), "Flip de la máscara incorrecto."
    assert mask.dtype == torch.int64, "La máscara debe ser int64."


def test_segmentation_no_transform() -> None:
    """Con probabilidades 0 la muestra no cambia."""

    ds = m.SyntheticSegmentationDataset(hflip_prob=0.0, vflip_prob=0.0)
    image, mask = ds[1]
    assert torch.equal(image, ds.images[1]), "No debería transformarse la imagen."
    assert torch.equal(mask, ds.masks[1]), "No debería transformarse la máscara."


def test_segmentation_same_transform_for_image_and_mask() -> None:
    """Con p = 0.5 la máscara sigue siendo coherente con la imagen."""

    ds = m.SyntheticSegmentationDataset(
        num_samples=4, image_size=IMAGE_SIZE, num_classes=NUM_CLASSES
    )
    flipped = 0
    for _ in range(5):
        for idx in range(len(ds)):
            image, mask = ds[idx]
            expected = (image[0] * NUM_CLASSES).long().clamp(max=NUM_CLASSES - 1)
            assert torch.equal(mask, expected), "Imagen y máscara desalineadas."
            flipped += int(not torch.equal(image, ds.images[idx]))
    assert 0 < flipped < 20, "Con p = 0.5 deberían salir muestras con y sin flip."


# ---------------------------------------------------------------------------
# detection_collate_fn
# ---------------------------------------------------------------------------
def test_detection_collate_fn() -> None:
    """Apila imágenes y devuelve los targets como lista."""

    batch = [
        (torch.rand(3, 8, 8), {"boxes": torch.rand(n, 4), "labels": torch.ones(n)})
        for n in (1, 3, 0)
    ]
    images, targets = m.detection_collate_fn(batch)
    assert images.shape == (3, 3, 8, 8), "Las imágenes deben apilarse."
    assert isinstance(targets, list) and len(targets) == 3, "targets debe ser lista."
    assert [t["boxes"].shape[0] for t in targets] == [1, 3, 0], "Cajas mezcladas."


def test_detection_collate_fn_in_dataloader() -> None:
    """El ``default_collate`` falla con nº variable de cajas; el nuestro no."""

    class _DetDS(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, i: int) -> tuple[torch.Tensor, dict]:
            return torch.rand(3, 4, 4), {"boxes": torch.rand(i + 1, 4)}

    with pytest.raises(RuntimeError):
        next(iter(DataLoader(_DetDS(), batch_size=4)))
    images, targets = next(
        iter(DataLoader(_DetDS(), batch_size=4, collate_fn=m.detection_collate_fn))
    )
    assert images.shape == (4, 3, 4, 4), "Forma de imágenes incorrecta."
    assert targets[3]["boxes"].shape == (4, 4), "Target de la muestra 3 incorrecto."


# ---------------------------------------------------------------------------
# compute_mean_std
# ---------------------------------------------------------------------------
def test_compute_mean_std_matches_global() -> None:
    """Coincide con la media/desviación de todo el tensor (varios lotes)."""

    x = torch.rand(NUM_SAMPLES, 3, 5, 5) * torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1)
    loader = DataLoader(TensorDataset(x, torch.zeros(NUM_SAMPLES)), batch_size=3)
    mean, std = m.compute_mean_std(loader)
    assert mean.shape == (3,) and std.shape == (3,), "Deben ser vectores [C]."
    assert_close(mean, x.mean(dim=(0, 2, 3)))
    assert_close(std, x.std(dim=(0, 2, 3), unbiased=False), atol=1e-4)


def test_compute_mean_std_by_hand() -> None:
    """Ejemplo a mano: un canal con valores 0 y 2 → media 1, std 1."""

    batch = torch.tensor([0.0, 2.0, 0.0, 2.0]).view(1, 1, 2, 2)
    mean, std = m.compute_mean_std([batch])
    assert torch.isclose(mean, torch.tensor([1.0])).all(), "Media incorrecta."
    assert torch.isclose(std, torch.tensor([1.0])).all(), "Desviación incorrecta."


# ---------------------------------------------------------------------------
# make_weighted_sampler
# ---------------------------------------------------------------------------
def test_weighted_sampler_weights() -> None:
    """Pesos = 1 / frecuencia de la clase."""

    labels = torch.tensor([0, 0, 0, 1])
    sampler = m.make_weighted_sampler(labels)
    assert isinstance(sampler, WeightedRandomSampler), "Debe ser WeightedRandomSampler."
    expected = torch.tensor([1 / 3, 1 / 3, 1 / 3, 1.0], dtype=torch.float64)
    assert torch.allclose(sampler.weights.double(), expected), "Pesos incorrectos."
    assert sampler.replacement, "Debe muestrear con reemplazo."
    assert len(sampler) == 4, "num_samples por defecto = len(labels)."


def test_weighted_sampler_balances_classes() -> None:
    """Con un dataset 90/10 el sampler devuelve ~50/50."""

    labels = torch.tensor([0] * 90 + [1] * 10)
    g = torch.Generator().manual_seed(0)
    sampler = m.make_weighted_sampler(labels, num_samples=4000, generator=g)
    drawn = labels[torch.tensor(list(sampler))]
    frac = drawn.float().mean().item()
    assert 0.45 < frac < 0.55, f"Clase minoritaria al {frac:.2f}, se esperaba ~0.5."


# ---------------------------------------------------------------------------
# denormalize
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 4, 4)])
def test_denormalize_inverts_normalize(shape: tuple[int, ...]) -> None:
    """``denormalize(Normalize(x)) == x`` para imagen y para lote."""

    x = torch.rand(shape)
    normalized = v2.Normalize(m.IMAGENET_MEAN, m.IMAGENET_STD)(x)
    assert_close(m.denormalize(normalized), x)


def test_denormalize_by_hand() -> None:
    """Un píxel 0 vuelve a la media del canal."""

    out = m.denormalize(torch.zeros(3, 1, 1), mean=(0.5, 0.2, 0.1), std=(2, 2, 2))
    assert_close(out.flatten(), torch.tensor([0.5, 0.2, 0.1]))


# ---------------------------------------------------------------------------
# train_val_split y make_dataloader
# ---------------------------------------------------------------------------
def test_train_val_split_sizes_and_disjoint() -> None:
    """Tamaños correctos, índices disjuntos y que cubren todo."""

    ds = TensorDataset(torch.arange(NUM_SAMPLES))
    train, val = m.train_val_split(ds, val_fraction=0.3, seed=1)
    assert (len(train), len(val)) == (7, 3), "Tamaños incorrectos."
    idx = set(train.indices) | set(val.indices)
    assert idx == set(range(NUM_SAMPLES)), "Los subconjuntos deben cubrir el dataset."
    assert not set(train.indices) & set(val.indices), "No deben solaparse."


def test_train_val_split_reproducible() -> None:
    """Misma semilla → mismo split, aunque cambie la semilla global."""

    ds = TensorDataset(torch.arange(50))
    _, val1 = m.train_val_split(ds, 0.2, seed=3)
    torch.manual_seed(1234)
    _, val2 = m.train_val_split(ds, 0.2, seed=3)
    _, val3 = m.train_val_split(ds, 0.2, seed=4)
    assert list(val1.indices) == list(val2.indices), "El split no es reproducible."
    assert list(val1.indices) != list(val3.indices), "La semilla debe influir."


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_train_val_split_raises(fraction: float) -> None:
    """Fracciones fuera de (0, 1) → ``ValueError``."""

    with pytest.raises(ValueError):
        m.train_val_split(TensorDataset(torch.arange(4)), fraction)


def test_make_dataloader_reproducible() -> None:
    """Dos loaders con la misma semilla barajan igual."""

    ds = TensorDataset(torch.arange(20))
    order = [
        torch.cat([b[0] for b in m.make_dataloader(ds, batch_size=5, seed=7)])
        for _ in range(2)
    ]
    other = torch.cat([b[0] for b in m.make_dataloader(ds, batch_size=5, seed=8)])
    assert torch.equal(order[0], order[1]), "Mismo seed → mismo orden."
    assert not torch.equal(order[0], other), "Distinto seed → distinto orden."
    assert not torch.equal(order[0], torch.arange(20)), "Debería barajar."


def test_make_dataloader_options() -> None:
    """Respeta ``drop_last``, ``shuffle`` y ``worker_init_fn``."""

    loader = m.make_dataloader(
        TensorDataset(torch.arange(10)), batch_size=4, shuffle=False, drop_last=True
    )
    assert len(loader) == 2, "drop_last=True → 2 lotes."
    assert loader.worker_init_fn is m.seed_worker, "Falta worker_init_fn."
    assert torch.equal(next(iter(loader))[0], torch.arange(4)), "No debe barajar."


# ---------------------------------------------------------------------------
# YOLODataset
# ---------------------------------------------------------------------------
S, C = 7, 3


def test_yolo_encode_single_box() -> None:
    """Valores calculados a mano para una caja."""

    ds = m.YOLODataset(torch.zeros(1, 3, 14, 14), [], split_size=S, num_classes=C)
    # centro (0.5, 0.25): columna j = 3, fila i = 1; x_cell = 0.5, y_cell = 0.75
    target = ds.encode_target(torch.tensor([[2, 0.5, 0.25, 0.2, 0.4]]))
    assert target.shape == (S, S, C + 5), "Forma de la matriz objetivo incorrecta."
    expected = torch.tensor([0, 0, 1, 1, 0.5, 0.75, 1.4, 2.8])
    assert torch.allclose(target[1, 3], expected, atol=1e-5), "Celda (1, 3) mal."
    assert target.sum(dim=-1).count_nonzero() == 1, "Solo una celda con objeto."


def test_yolo_encode_collision_and_border() -> None:
    """Dos objetos en la misma celda → se queda el primero; x = 1 → última celda."""

    ds = m.YOLODataset(torch.zeros(1, 3, 7, 7), [], split_size=S, num_classes=C)
    boxes = torch.tensor(
        [[0, 0.05, 0.05, 0.1, 0.1], [1, 0.1, 0.1, 0.1, 0.1], [2, 1.0, 1.0, 0.1, 0.1]]
    )
    target = ds.encode_target(boxes)
    assert target[0, 0, 0] == 1 and target[0, 0, 1] == 0, "Debe quedar la 1.ª caja."
    assert target[S - 1, S - 1, C] == 1, "x = y = 1 debe ir a la celda (S-1, S-1)."
    assert target[..., C].sum() == 2, "Debe haber 2 celdas con objeto."
    assert (target[..., C + 1 : C + 3] < 1 + 1e-6).all(), "x_cell, y_cell ≤ 1."


def test_yolo_dataset_getitem_and_loader() -> None:
    """``__getitem__`` y ``DataLoader`` devuelven el lote [B, S, S, C + 5]."""

    images = torch.rand(4, 3, 14, 14)
    ann = [torch.tensor([[i % C, 0.3, 0.6, 0.2, 0.2]]) for i in range(4)]
    ds = m.YOLODataset(images, ann, split_size=S, num_classes=C)
    assert len(ds) == 4, "len incorrecto."
    image, target = ds[2]
    assert torch.equal(image, images[2]), "Imagen incorrecta."
    assert target[4, 2, 2] == 1, "Clase de la muestra 2 en la celda (4, 2)."
    imgs, targets = next(iter(DataLoader(ds, batch_size=4)))
    assert targets.shape == (4, S, S, C + 5), "Forma del lote de targets incorrecta."


# ---------------------------------------------------------------------------
# make_detection_transform (transforms v2)
# ---------------------------------------------------------------------------
def test_detection_transform_flips_boxes_and_mask() -> None:
    """Con p = 1 la caja y la máscara se reflejan igual que la imagen."""

    transform = m.make_detection_transform(size=8, hflip_prob=1.0)
    image = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
    boxes = tv_tensors.BoundingBoxes(
        torch.tensor([[0.0, 0.0, 2.0, 4.0]]), format="XYXY", canvas_size=(8, 8)
    )
    mask = tv_tensors.Mask(torch.zeros(8, 8, dtype=torch.uint8))
    mask[:, :2] = 1
    out_img, out_boxes, out_mask = transform(image, boxes, mask)
    assert out_img.dtype == torch.float32, "La imagen debe pasar a float32."
    assert out_img.max() <= 1.0, "ToDtype(scale=True) debe escalar a [0, 1]."
    assert_close(out_img, image.flip(-1).float() / 255)
    assert_close(out_boxes, torch.tensor([[6.0, 0.0, 8.0, 4.0]]))
    assert out_mask[:, 6:].all() and not out_mask[:, :6].any(), "Máscara mal."


def test_detection_transform_resizes_boxes() -> None:
    """El ``Resize`` escala también las cajas."""

    transform = m.make_detection_transform(size=4, hflip_prob=0.0)
    boxes = tv_tensors.BoundingBoxes(
        torch.tensor([[2.0, 2.0, 6.0, 8.0]]), format="XYXY", canvas_size=(8, 8)
    )
    out_img, out_boxes = transform(torch.rand(3, 8, 8), boxes)
    assert out_img.shape == (3, 4, 4), "Resize incorrecto."
    assert_close(out_boxes, torch.tensor([[1.0, 1.0, 3.0, 4.0]]))
