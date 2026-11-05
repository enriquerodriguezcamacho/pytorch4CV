"""
Tests de los ejercicios de los temas 8 y 9 de los apuntes.
"""

import pytest
import torch
import torch.nn.functional as F
from helpers import assert_close, load

t = load("cv/08_attention_transformers/apuntes.py")
s = load("cv/09_audio/apuntes.py")


def test_batched_attention_matches_torch() -> None:
    """Coincide con ``F.scaled_dot_product_attention`` y tiene la forma pedida."""

    q, k, v = torch.randn(2, 4, 3, 8), torch.randn(2, 4, 5, 8), torch.randn(2, 4, 5, 6)
    out = t.batched_attention(q, k, v)
    assert out.shape == (2, 4, 3, 6)
    assert_close(out, F.scaled_dot_product_attention(q, k, v))


def test_batched_attention_additive_mask() -> None:
    """Las posiciones con ``-inf`` no se atienden; 0 no cambia nada."""

    q, k = torch.randn(1, 1, 2, 4), torch.randn(1, 1, 3, 4)
    v = torch.tensor([[[[1.0], [2.0], [3.0]]]])
    blocked = torch.tensor([0.0, float("-inf"), float("-inf")])
    assert_close(t.batched_attention(q, k, v, blocked), torch.ones(1, 1, 2, 1))
    zeros = torch.zeros(3)
    assert_close(t.batched_attention(q, k, v, zeros), t.batched_attention(q, k, v))


def test_deit_loss_hard() -> None:
    """Coincide con la fórmula de los apuntes."""

    cls, dist, teacher = torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3)
    y = torch.tensor([0, 1, 2, 0])
    expected = 0.3 * F.cross_entropy(cls, y) + 0.7 * F.cross_entropy(
        dist, teacher.argmax(dim=1)
    )
    assert_close(t.deit_loss(cls, dist, teacher, y, "hard", 0.3), expected)


def test_deit_loss_soft_and_teacher_is_fixed() -> None:
    """KL nula con distribuciones iguales y sin gradiente hacia el profesor."""

    logits = torch.randn(4, 3)
    teacher = logits.clone().requires_grad_(True)
    y = torch.tensor([0, 1, 2, 0])
    loss = t.deit_loss(logits, logits, teacher, y, "soft", 0.5)
    assert_close(loss, 0.5 * F.cross_entropy(logits, y))
    assert not loss.requires_grad
    with pytest.raises(ValueError):
        t.deit_loss(logits, logits, teacher, y, "otra")


def test_window_partition_shape_order_and_reverse() -> None:
    """Forma correcta, primera ventana correcta y reversibilidad exacta."""

    x = torch.arange(2 * 8 * 12 * 3.0).view(2, 8, 12, 3)
    windows = t.window_partition(x, 4)
    assert windows.shape == (2 * 8 * 12 // 16, 4, 4, 3)
    assert torch.equal(windows[0], x[0, :4, :4])
    assert torch.equal(windows[1], x[0, :4, 4:8])
    assert torch.equal(t.window_reverse(windows, 8, 12), x)


def test_duration_to_samples() -> None:
    """25 ms y 10 ms a 16 kHz son 400 y 160 muestras."""

    assert s.duration_to_samples(25, 16000) == 400
    assert s.duration_to_samples(10, 16000) == 160


def test_ctc_collapse() -> None:
    """El ejemplo del ejercicio da ``cat`` y el *blank* conserva letras dobles."""

    assert s.ctc_collapse(list("cc") + ["∅"] + list("aa") + ["∅"] + list("tt")) == "cat"
    assert s.ctc_collapse(["l", "∅", "l", "l"]) == "ll"
    assert s.ctc_collapse(["∅", "∅"]) == ""


def test_word_error_alignment_exercise() -> None:
    """Ejemplo del ejercicio: 2 ediciones sobre 4 palabras."""

    result = s.word_error_alignment("a small red car", "a red fast car")
    assert result.wer == pytest.approx(0.5)
    assert result.substitutions + result.deletions + result.insertions == 2


def test_word_error_alignment_breakdown() -> None:
    """Sustitución, borrado e inserción puros."""

    assert s.word_error_alignment("a b", "a c")[1:] == (1, 0, 0)
    assert s.word_error_alignment("a b c", "a c")[1:] == (0, 1, 0)
    assert s.word_error_alignment("a c", "a b c")[1:] == (0, 0, 1)
    assert s.word_error_alignment("a b", "a b").wer == 0
