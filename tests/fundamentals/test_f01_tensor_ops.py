"""
Tests de F01 · Métodos de tensores.
"""

import torch
from helpers import assert_close, load

m = load("fundamentals/01_tensors/tensor_ops.py")


# A. Creación


def test_zeros_ones_like_keeps_dtype() -> None:
    x = torch.tensor([[1, 2, 3]], dtype=torch.int16)
    zeros, ones = m.zeros_ones_like(x)
    assert zeros.shape == x.shape and zeros.dtype == torch.int16 and zeros.sum() == 0
    assert ones.shape == x.shape and ones.dtype == torch.int16 and ones.sum() == 3


def test_constant_image() -> None:
    image = m.constant_image(3, 2, 4, 7)
    assert image.shape == (3, 2, 4) and image.dtype == torch.float32
    assert (image == 7).all()


def test_identity_batch_does_not_share_memory() -> None:
    batch = m.identity_batch(2, 3)
    assert torch.equal(batch[1], torch.eye(3))
    batch[0, 0, 0] = 9
    assert batch[1, 0, 0] == 1


def test_even_numbers() -> None:
    assert m.even_numbers(4).tolist() == [0, 2, 4, 6]


def test_interpolation_steps() -> None:
    assert m.interpolation_steps(0.0, 1.0, 5).tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_gaussian_noise_like() -> None:
    x = torch.zeros(10000, dtype=torch.float64)
    noise = m.gaussian_noise_like(x, 2.0)
    assert noise.shape == x.shape and noise.dtype == torch.float64
    assert abs(noise.std().item() - 2.0) < 0.1 and abs(noise.mean().item()) < 0.1


def test_random_labels() -> None:
    labels = m.random_labels(100, 3)
    assert labels.shape == (100,) and labels.dtype == torch.int64
    assert labels.min() >= 0 and labels.max() < 3 and labels.unique().numel() == 3


def test_blank_like_keeps_dtype_with_new_shape() -> None:
    out = m.blank_like(torch.ones(2, dtype=torch.float64), (3, 4))
    assert out.shape == (3, 4) and out.dtype == torch.float64 and out.sum() == 0


def test_causal_mask() -> None:
    expected = [[True, False, False], [True, True, False], [True, True, True]]
    assert m.causal_mask(3).tolist() == expected


# B. Tipos de datos


def test_to_uint8_image() -> None:
    out = m.to_uint8_image(torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0]))
    assert out.dtype == torch.uint8
    assert out.tolist() == [0, 0, 128, 255, 255]


def test_mean_of_labels() -> None:
    out = m.mean_of_labels(torch.tensor([1, 2, 6]))
    assert out.dtype == torch.float32 and out.item() == 3.0


# C. Cambiar la forma


def test_add_batch_and_channel() -> None:
    assert m.add_batch_and_channel(torch.zeros(4, 5)).shape == (1, 1, 4, 5)


def test_flatten_per_sample() -> None:
    assert m.flatten_per_sample(torch.zeros(2, 3, 4, 5)).shape == (2, 60)


def test_hwc_batch_to_chw_moves_data() -> None:
    x = torch.rand(2, 4, 5, 3)
    out = m.hwc_batch_to_chw(x)
    assert out.shape == (2, 3, 4, 5)
    assert out[1, 2, 3, 4] == x[1, 3, 4, 2]


def test_remove_singletons() -> None:
    assert m.remove_singletons(torch.zeros(1, 3, 1, 4, 1)).shape == (3, 4)


def test_tile_image() -> None:
    x = torch.tensor([[[1, 2]]])
    assert m.tile_image(x, 2, 2).tolist() == [[[1, 2, 1, 2], [1, 2, 1, 2]]]


def test_expand_to_batch_shares_memory() -> None:
    x = torch.zeros(2, 3)
    out = m.expand_to_batch(x, 4)
    assert out.shape == (4, 2, 3)
    x[0, 0] = 5
    assert (out[:, 0, 0] == 5).all()


def test_repeat_each() -> None:
    assert m.repeat_each(torch.tensor([1, 2]), 2).tolist() == [1, 1, 2, 2]


def test_horizontal_flip() -> None:
    assert m.horizontal_flip(torch.tensor([[1, 2, 3]])).tolist() == [[3, 2, 1]]


def test_rotate_90_counterclockwise() -> None:
    assert m.rotate_90(torch.tensor([[1, 2], [3, 4]])).tolist() == [[2, 4], [1, 3]]
    assert m.rotate_90(torch.zeros(1, 2, 3)).shape == (1, 3, 2)


def test_cyclic_shift() -> None:
    x = torch.tensor([[1, 2], [3, 4]])
    assert m.cyclic_shift(x, 1).tolist() == [[4, 3], [2, 1]]


def test_frame_signal_1d() -> None:
    frames = m.frame_signal_1d(torch.arange(6), 3, 2)
    assert frames.tolist() == [[0, 1, 2], [2, 3, 4]]


# D. Unir y separar


def test_stack_images() -> None:
    assert m.stack_images([torch.zeros(3, 4, 5)] * 2).shape == (2, 3, 4, 5)


def test_concat_channels() -> None:
    out = m.concat_channels(torch.zeros(2, 3, 4, 4), torch.ones(2, 5, 4, 4))
    assert out.shape == (2, 8, 4, 4) and out.sum() == 2 * 5 * 16


def test_xywh_to_xyxy() -> None:
    boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert m.xywh_to_xyxy(boxes).tolist() == [[1.0, 2.0, 4.0, 6.0]]


def test_chunk_heads() -> None:
    parts = m.chunk_heads(torch.zeros(2, 8), 4)
    assert len(parts) == 4 and all(part.shape == (2, 2) for part in parts)


def test_split_channels() -> None:
    parts = m.split_channels(torch.zeros(2, 6, 4), [1, 2, 3])
    assert [part.shape[1] for part in parts] == [1, 2, 3]


# E. Indexado y selección


def test_select_rows() -> None:
    x = torch.tensor([[1], [2], [3]])
    assert m.select_rows(x, torch.tensor([2, 0])).tolist() == [[3], [1]]


def test_replace_negatives_does_not_modify_input() -> None:
    x = torch.tensor([-1.0, 2.0])
    assert m.replace_negatives(x, 0.0).tolist() == [0.0, 2.0]
    assert x[0] == -1


def test_fill_masked_does_not_modify_input() -> None:
    x = torch.tensor([1.0, 2.0])
    out = m.fill_masked(x, torch.tensor([True, False]), -5.0)
    assert out.tolist() == [-5.0, 2.0] and x[0] == 1


def test_true_coordinates() -> None:
    mask = torch.tensor([[False, True], [True, False]])
    assert m.true_coordinates(mask).tolist() == [[0, 1], [1, 0]]


def test_sort_by_score() -> None:
    boxes = torch.tensor([[0.0] * 4, [1.0] * 4, [2.0] * 4])
    out = m.sort_by_score(boxes, torch.tensor([0.1, 0.9, 0.5]))
    assert out[:, 0].tolist() == [1.0, 2.0, 0.0]


def test_argmax_2d() -> None:
    assert m.argmax_2d(torch.tensor([[0, 1, 2], [3, 9, 4]])) == (1, 1)


def test_labels_to_one_hot() -> None:
    out = m.labels_to_one_hot(torch.tensor([2, 0]), 3)
    assert out.dtype == torch.float32
    assert out.tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]


# F. Reducciones y matemáticas


def test_channel_mean_std() -> None:
    x = torch.rand(4, 3, 5, 5)
    mean, std = m.channel_mean_std(x)
    assert mean.shape == (3,) and std.shape == (3,)
    assert_close(mean[1], x[:, 1].mean())
    assert_close(std[2], x[:, 2].std())


def test_minmax_per_image_is_independent() -> None:
    x = torch.stack([torch.rand(1, 3, 3) * 10, torch.rand(1, 3, 3) - 5])
    out = m.minmax_per_image(x)
    assert_close(out.amin(dim=(1, 2, 3)), torch.zeros(2))
    assert_close(out.amax(dim=(1, 2, 3)), torch.ones(2))


def test_minmax_per_image_constant_image() -> None:
    assert (m.minmax_per_image(torch.full((1, 1, 2, 2), 3.0)) == 0).all()


def test_l2_normalize() -> None:
    out = m.l2_normalize(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
    assert_close(out, torch.tensor([[0.6, 0.8], [0.0, 0.0]]))


def test_cosine_similarity_matrix() -> None:
    a, b = torch.randn(3, 4), torch.randn(2, 4)
    expected = torch.cosine_similarity(a[:, None], b[None], dim=-1)
    assert_close(m.cosine_similarity_matrix(a, b), expected)


def test_running_sum() -> None:
    assert m.running_sum(torch.tensor([1, 2, 3])).tolist() == [1, 3, 6]


def test_count_per_class_includes_missing() -> None:
    assert m.count_per_class(torch.tensor([0, 0, 2]), 4).tolist() == [2, 0, 1, 0]


def test_unique_with_counts() -> None:
    values, counts = m.unique_with_counts(torch.tensor([3, 1, 3, 3]))
    assert values.tolist() == [1, 3] and counts.tolist() == [1, 3]


def test_rows_all_any() -> None:
    mask = torch.tensor([[True, True], [True, False], [False, False]])
    every, some = m.rows_all_any(mask)
    assert every.tolist() == [True, False, False]
    assert some.tolist() == [True, True, False]


def test_clip_boxes() -> None:
    boxes = torch.tensor([[-1.0, -2.0, 50.0, 60.0]])
    assert m.clip_boxes(boxes, 20, 30).tolist() == [[0.0, 0.0, 30.0, 20.0]]


def test_blend() -> None:
    a, b = torch.tensor([0.0, 10.0]), torch.tensor([10.0, 20.0])
    assert m.blend(a, b, 0.25).tolist() == [2.5, 12.5]


# G. Memoria y conversión


def test_independent_copy() -> None:
    x = torch.ones(2, requires_grad=True)
    copy = m.independent_copy(x)
    assert not copy.requires_grad
    copy[0] = 5
    assert x[0] == 1


def test_to_python_list() -> None:
    out = m.to_python_list(torch.tensor([[1, 2]]))
    assert out == [[1, 2]] and isinstance(out, list)


def test_nearly_equal() -> None:
    a = torch.tensor([1.0, 2.0])
    assert m.nearly_equal(a, a + 0.01, 0.1)
    assert not m.nearly_equal(a, a + 0.5, 0.1)
