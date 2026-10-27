"""
Tests de F01 · Tensores.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from helpers import assert_close, load

m = load("fundamentals/01_tensors/tensors.py")


def test_to_chw_tensor() -> None:
    img = np.random.randint(0, 256, size=(8, 6, 3), dtype=np.uint8)
    out = m.to_chw_tensor(img)
    assert out.shape == (3, 8, 6)
    assert out.dtype == torch.float32
    assert_close(out[1, 2, 3], torch.tensor(img[2, 3, 1] / 255.0, dtype=torch.float32))


def test_normalize_per_channel() -> None:
    x = torch.rand(4, 3, 5, 5)
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    out = m.normalize_per_channel(x, mean, std)
    assert out.shape == x.shape
    assert_close(out[:, 2], (x[:, 2] - 0.406) / 0.225)


def test_pairwise_sq_distances() -> None:
    a, b = torch.randn(7, 5), torch.randn(4, 5)
    out = m.pairwise_sq_distances(a, b)
    assert out.shape == (7, 4)
    assert_close(out, torch.cdist(a, b) ** 2, atol=1e-4)
    assert (m.pairwise_sq_distances(a, a).diagonal() < 1e-4).all()


def test_select_by_index() -> None:
    logits = torch.randn(6, 10)
    labels = torch.randint(0, 10, (6,))
    out = m.select_by_index(logits, labels)
    assert out.shape == (6,)
    assert_close(out, logits[torch.arange(6), labels])


def test_one_hot_mask() -> None:
    mask = torch.randint(0, 4, (2, 5, 6))
    out = m.one_hot_mask(mask, 4)
    assert out.shape == (2, 4, 5, 6)
    assert out.dtype == torch.float32
    assert_close(out, F.one_hot(mask, 4).permute(0, 3, 1, 2).float())


@pytest.mark.parametrize("p", [2, 4])
def test_image_to_patches(p: int) -> None:
    x = torch.randn(2, 3, 8, 8)
    out = m.image_to_patches(x, p)
    n = (8 // p) ** 2
    assert out.shape == (2, n, 3 * p * p)
    # el segundo parche es el de la fila 0, columna 1
    assert_close(out[1, 1], x[1, :, :p, p : 2 * p].reshape(-1))
    # equivalente a unfold con stride = kernel
    ref = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2)
    assert_close(out, ref)


def test_batched_gram_matrix() -> None:
    x = torch.randn(2, 4, 3, 5)
    out = m.batched_gram_matrix(x)
    assert out.shape == (2, 4, 4)
    f = x[0].reshape(4, -1)
    assert_close(out[0], f @ f.T / (4 * 3 * 5))
    assert_close(out, out.transpose(1, 2))


def test_topk_accuracy() -> None:
    logits = torch.tensor([[0.1, 0.5, 0.4], [0.9, 0.05, 0.05], [0.2, 0.3, 0.5]])
    labels = torch.tensor([2, 0, 0])
    assert m.topk_accuracy(logits, labels, 1) == pytest.approx(1 / 3)
    assert m.topk_accuracy(logits, labels, 2) == pytest.approx(2 / 3)
    assert m.topk_accuracy(logits, labels, 3) == pytest.approx(1.0)


def test_masked_mean() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[True, False, True], [False, False, False]])
    out = m.masked_mean(x, mask, dim=1)
    assert out.shape == (2,)
    assert_close(out, torch.tensor([2.0, 0.0]))  # fila sin válidos → 0
    full = torch.ones_like(x, dtype=torch.bool)
    assert_close(m.masked_mean(x, full, dim=0), x.mean(0))


@pytest.mark.parametrize("p", [2, 4])
def test_patches_to_image_roundtrip(p: int) -> None:
    x = torch.randn(2, 3, 8, 12)
    patches = m.image_to_patches(x, p)
    out = m.patches_to_image(patches, p, 8, 12)
    assert out.shape == x.shape
    assert torch.equal(out, x)
    ref = F.fold(patches.transpose(1, 2), (8, 12), kernel_size=p, stride=p)
    assert_close(out, ref)


def test_class_histogram() -> None:
    mask = torch.tensor([[[0, 1], [1, 1]], [[2, 2], [2, 0]]])
    out = m.class_histogram(mask, 4)
    assert out.dtype == torch.int64
    assert out.tolist() == [[1, 3, 0, 0], [1, 0, 3, 0]]
    big = torch.randint(0, 5, (3, 7, 9))
    expected = torch.stack([torch.bincount(b.flatten(), minlength=5) for b in big])
    assert torch.equal(m.class_histogram(big, 5), expected)


def test_grid_coordinates() -> None:
    out = m.grid_coordinates(2, 3)
    assert out.shape == (2, 3, 2)
    assert out.dtype == torch.float32
    assert out[0, 0].tolist() == [0.5, 0.5]
    assert out[1, 2].tolist() == [2.5, 1.5], "Orden (x, y): columna primero."


def test_mask_to_box() -> None:
    mask = torch.zeros(2, 5, 6, dtype=torch.bool)
    mask[0, 1:3, 2:5] = True
    mask[1, 4, 0] = True
    out = m.mask_to_box(mask)
    assert out.dtype == torch.float32
    assert out.tolist() == [[2.0, 1.0, 4.0, 2.0], [0.0, 4.0, 0.0, 4.0]]
    from torchvision.ops import masks_to_boxes

    rand = torch.rand(4, 8, 8) > 0.7
    rand[:, 3, 3] = True
    assert_close(m.mask_to_box(rand), masks_to_boxes(rand))


def test_batched_index_select() -> None:
    feats = torch.randn(2, 6, 4)
    idx = torch.tensor([[5, 0], [2, 2]])
    out = m.batched_index_select(feats, idx)
    assert out.shape == (2, 2, 4)
    assert_close(out[0, 0], feats[0, 5])
    assert_close(out[1, 1], feats[1, 2])
    top = feats.norm(dim=-1).topk(3, dim=1).indices
    assert_close(m.batched_index_select(feats, top), feats[torch.arange(2)[:, None], top])
