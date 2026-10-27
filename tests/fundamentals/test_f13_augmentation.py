"""
Tests de F13 · Aumento de datos.
"""

import pytest
import torch
import torch.nn.functional as F
from helpers import DummyCNN, assert_close, load
from torch import nn

m = load("fundamentals/13_data_augmentation/augmentation.py")

TF = pytest.importorskip("torchvision.transforms.v2.functional")

C, H, W = 3, 10, 12


def _boxes() -> torch.Tensor:
    return torch.tensor([[1.0, 2.0, 5.0, 6.0], [7.0, 0.0, 12.0, 4.0]])


# --------------------------------------------------------------------------- #
# Geometría                                                                   #
# --------------------------------------------------------------------------- #


def test_hflip_boxes_hand_values() -> None:
    out = m.hflip_boxes(_boxes(), W)
    assert_close(out, torch.tensor([[7.0, 2.0, 11.0, 6.0], [0.0, 0.0, 5.0, 4.0]]))
    assert_close(m.hflip_boxes(out, W), _boxes())  # voltear dos veces es la identidad


def test_hflip_matches_torchvision() -> None:
    from torchvision import tv_tensors

    image = torch.rand(C, H, W)
    boxes = tv_tensors.BoundingBoxes(_boxes(), format="XYXY", canvas_size=(H, W))
    assert_close(m.hflip(image)[0], TF.horizontal_flip(image))
    assert_close(m.hflip_boxes(_boxes(), W), TF.horizontal_flip(boxes).as_subclass(torch.Tensor))


def test_hflip_target_consistency() -> None:
    """Un píxel marcado en la imagen sigue dentro de su caja y de su máscara tras voltear."""

    image = torch.zeros(1, H, W)
    image[0, 3, 2] = 1.0
    mask = torch.zeros(H, W, dtype=torch.long)
    mask[3, 2] = 4
    target = {"boxes": torch.tensor([[2.0, 3.0, 3.0, 4.0]]), "masks": mask, "labels": torch.tensor([4])}
    out_img, out_t = m.hflip(image, target)
    row, col = (out_img[0] == 1).nonzero()[0].tolist()
    assert out_t["masks"][row, col] == 4
    x1, y1, x2, y2 = out_t["boxes"][0].tolist()
    assert x1 <= col < x2 and y1 <= row < y2
    assert out_t["labels"] is target["labels"]
    assert torch.equal(target["masks"], mask), "No modifiques el target original."
    assert m.hflip(image, None)[1] is None


def test_crop_with_boxes() -> None:
    image = torch.arange(C * H * W, dtype=torch.float32).view(C, H, W)
    boxes = torch.tensor([[1.0, 1.0, 4.0, 4.0], [5.0, 5.0, 9.0, 9.0], [0.0, 0.0, 1.0, 1.0]])
    crop, kept, keep = m.crop_with_boxes(image, boxes, top=2, left=2, height=5, width=6)
    assert crop.shape == (C, 5, 6)
    assert torch.equal(crop, image[:, 2:7, 2:8])
    assert keep.tolist() == [True, True, False], "La tercera caja queda fuera."
    assert_close(kept, torch.tensor([[0.0, 0.0, 2.0, 2.0], [3.0, 3.0, 6.0, 5.0]]))


def test_crop_with_boxes_min_area() -> None:
    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0]])  # tras el recorte queda 2×2 = 25 %
    _, kept, keep = m.crop_with_boxes(torch.rand(C, H, W), boxes, 2, 2, 5, 5, min_area_ratio=0.5)
    assert keep.tolist() == [False] and kept.shape == (0, 4)
    _, kept, _ = m.crop_with_boxes(torch.rand(C, H, W), boxes, 2, 2, 5, 5, min_area_ratio=0.25)
    assert kept.shape == (1, 4)


def test_resize_with_boxes_matches_torchvision() -> None:
    from torchvision import tv_tensors

    image = torch.rand(C, H, W)
    out, boxes = m.resize_with_boxes(image, _boxes(), (20, 6))
    assert out.shape == (C, 20, 6)
    ref = TF.resize(image, [20, 6], antialias=False)
    assert_close(out, ref, atol=1e-5)
    tv_boxes = tv_tensors.BoundingBoxes(_boxes(), format="XYXY", canvas_size=(H, W))
    assert_close(boxes, TF.resize(tv_boxes, [20, 6]).as_subclass(torch.Tensor), atol=1e-4)


def test_resize_mask_keeps_labels() -> None:
    mask = torch.randint(0, 3, (H, W))
    out = m.resize_mask(mask, (25, 7))
    assert out.shape == (25, 7) and out.dtype == torch.int64
    assert set(out.unique().tolist()) <= set(mask.unique().tolist()), "No inventa clases."
    assert torch.equal(m.resize_mask(mask, (2 * H, 2 * W))[::2, ::2], mask)


# --------------------------------------------------------------------------- #
# Color                                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("factor", [0.0, 0.5, 1.0, 1.7])
def test_brightness_and_contrast_match_torchvision(factor: float) -> None:
    image = torch.rand(2, C, H, W)
    assert_close(m.adjust_brightness(image, factor), TF.adjust_brightness(image, factor))
    assert_close(m.adjust_contrast(image, factor), TF.adjust_contrast(image, factor), atol=1e-5)


def test_contrast_zero_is_uniform_gray() -> None:
    out = m.adjust_contrast(torch.rand(C, H, W), 0.0)
    assert_close(out, out.flatten()[0].expand_as(out))


def test_cutout() -> None:
    image = torch.ones(C, H, W)
    out = m.cutout(image, top=8, left=10, size=4, value=0.5)
    assert (out[:, 8:, 10:] == 0.5).all()
    assert (out == 0.5).sum() == C * 2 * 2, "Se recorta al borde."
    assert image.eq(1).all(), "Trabaja sobre una copia."


# --------------------------------------------------------------------------- #
# Composición y TTA                                                           #
# --------------------------------------------------------------------------- #


def test_random_horizontal_flip_probability_and_seed() -> None:
    image = torch.rand(C, H, W)
    assert torch.equal(m.RandomHorizontalFlip(1.0)(image)[0], image.flip(-1))
    assert torch.equal(m.RandomHorizontalFlip(0.0)(image)[0], image)
    torch.manual_seed(7)
    out, _ = m.RandomHorizontalFlip(0.5)(image)
    torch.manual_seed(7)
    expected = image.flip(-1) if torch.rand(1).item() < 0.5 else image
    assert torch.equal(out, expected)
    flips = sum(not torch.equal(m.RandomHorizontalFlip(0.5)(image)[0], image) for _ in range(200))
    assert 70 < flips < 130


def test_paired_compose_order() -> None:
    calls: list[str] = []

    def first(img: torch.Tensor, t: dict | None) -> tuple:
        calls.append("first")
        return img + 1, t

    def second(img: torch.Tensor, t: dict | None) -> tuple:
        calls.append("second")
        return img * 2, t

    out, target = m.PairedCompose([first, second])(torch.zeros(1, 2, 2), {"a": torch.ones(1)})
    assert calls == ["first", "second"]
    assert torch.equal(out, torch.full((1, 2, 2), 2.0))
    assert "a" in target
    image = torch.rand(C, H, W)
    flip_twice = m.PairedCompose([m.RandomHorizontalFlip(1.0), m.RandomHorizontalFlip(1.0)])
    assert torch.equal(flip_twice(image, {"boxes": _boxes()})[1]["boxes"], _boxes())


def test_tta_predict() -> None:
    model = DummyCNN(4)
    model.train()
    x = torch.rand(2, C, H, W)
    out = m.tta_predict(model, x)
    assert not model.training, "Pasa a eval."
    expected = (F.softmax(model(x), 1) + F.softmax(model(x.flip(-1)), 1)) / 2
    assert_close(out, expected)
    assert_close(out.sum(1), torch.ones(2))


def test_tta_segmentation_unflips() -> None:
    """Con un modelo equivariante al volteo, la TTA coincide con la predicción normal."""

    model = nn.Conv2d(C, 3, kernel_size=1)
    x = torch.rand(2, C, H, W)
    out = m.tta_segmentation(model, x)
    assert out.shape == (2, 3, H, W)
    assert_close(out, F.softmax(model(x), 1), atol=1e-6)
    shifty = nn.Conv2d(C, 3, kernel_size=(1, 3), padding=(0, 1))
    expected = (F.softmax(shifty(x), 1) + F.softmax(shifty(x.flip(-1)), 1).flip(-1)) / 2
    assert_close(m.tta_segmentation(shifty, x), expected)
