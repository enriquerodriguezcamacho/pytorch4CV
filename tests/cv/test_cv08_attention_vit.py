"""
Tests de CV08 · Atención y Transformers: Nadaraya-Watson, atención, MHA, PE, encoder y ViT.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from helpers import assert_close, load
from torch import nn

a = load("cv/08_attention_transformers/attention.py")
vit = load("cv/08_attention_transformers/vit.py")

BATCH, LENGTH, D_MODEL, HEADS = 2, 5, 16, 4


# --------------------------------------------------------------------------- #
# Nadaraya-Watson                                                             #
# --------------------------------------------------------------------------- #


def test_nadaraya_watson_hand_values() -> None:
    keys = torch.tensor([0.0, 1.0, 3.0])
    values = torch.tensor([1.0, 3.0, 10.0])
    box = a.nadaraya_watson(torch.tensor([0.5, 3.0]), keys, values, "boxcar", 1.0)
    assert_close(box, torch.tensor([2.0, 10.0]), atol=1e-6)  # (1 + 3) / 2 y solo el 3
    epa = a.nadaraya_watson(torch.tensor([0.25]), keys, values, "epanechnikov", 1.0)
    assert_close(epa, torch.tensor([(0.75 * 1 + 0.25 * 3) / 1.0]), atol=1e-6)


def test_nadaraya_watson_gaussian() -> None:
    keys, values = torch.randn(10), torch.randn(10)
    queries = torch.randn(4)
    out = a.nadaraya_watson(queries, keys, values, "gaussian", bandwidth=0.5)
    weights = torch.softmax(-((queries[:, None] - keys[None]) / 0.5) ** 2 / 2, dim=1)
    assert out.shape == (4,), "Forma incorrecta."
    assert_close(out, weights @ values, atol=1e-5)
    grid = torch.arange(5.0)
    narrow = a.nadaraya_watson(grid, grid, values[:5], "gaussian", bandwidth=0.1)
    assert_close(narrow, values[:5], atol=1e-4)  # con h → 0 interpola los puntos


def test_nadaraya_watson_empty_window_and_invalid() -> None:
    out = a.nadaraya_watson(torch.tensor([100.0]), torch.zeros(3), torch.ones(3), "boxcar")
    assert out.item() == 0.0
    with pytest.raises(ValueError):
        a.nadaraya_watson(torch.zeros(1), torch.zeros(1), torch.zeros(1), "cosine")


# --------------------------------------------------------------------------- #
# Atención                                                                    #
# --------------------------------------------------------------------------- #


def test_causal_mask() -> None:
    mask = a.causal_mask(3)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[True, False, False], [True, True, False], [True, True, True]]


def test_sdpa_matches_torch() -> None:
    q, k, v = torch.randn(BATCH, 3, LENGTH, 8), torch.randn(BATCH, 3, 7, 8), torch.randn(BATCH, 3, 7, 6)
    out, weights = a.scaled_dot_product_attention(q, k, v)
    assert out.shape == (BATCH, 3, LENGTH, 6) and weights.shape == (BATCH, 3, LENGTH, 7)
    assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5)
    assert_close(weights.sum(-1), torch.ones(BATCH, 3, LENGTH))


def test_sdpa_causal_mask() -> None:
    q = k = v = torch.randn(BATCH, LENGTH, 8)
    mask = a.causal_mask(LENGTH)
    out, weights = a.scaled_dot_product_attention(q, k, v, mask)
    assert (weights.triu(1) == 0).all(), "No se atiende al futuro."
    assert_close(out, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5)
    assert_close(out[:, 0], v[:, 0])  # la primera posición solo se ve a sí misma


def test_sdpa_scaling_hand_value() -> None:
    """q·k = 4 con d = 4 → score 2; con dos keys (2, 0) los pesos son softmax([2, 0])."""

    q = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
    k = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])
    v = torch.tensor([[[1.0], [0.0]]])
    out, weights = a.scaled_dot_product_attention(q, k, v)
    p = math.exp(2) / (math.exp(2) + 1)
    assert_close(weights, torch.tensor([[[p, 1 - p]]]))
    assert_close(out, torch.tensor([[[p]]]))


def test_mha_split_merge_roundtrip() -> None:
    mha = a.MultiHeadAttention(D_MODEL, HEADS)
    x = torch.randn(BATCH, LENGTH, D_MODEL)
    split = mha._split_heads(x)
    assert split.shape == (BATCH, HEADS, LENGTH, D_MODEL // HEADS)
    assert_close(split[:, 1], x[..., 4:8])  # la cabeza 1 son las features 4..7
    assert torch.equal(mha._merge_heads(split), x)


def test_mha_matches_torch() -> None:
    ours = a.MultiHeadAttention(D_MODEL, HEADS)
    ref = nn.MultiheadAttention(D_MODEL, HEADS, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([ours.w_q.weight, ours.w_k.weight, ours.w_v.weight]))
        ref.in_proj_bias.copy_(torch.cat([ours.w_q.bias, ours.w_k.bias, ours.w_v.bias]))
        ref.out_proj.weight.copy_(ours.w_o.weight)
        ref.out_proj.bias.copy_(ours.w_o.bias)
    q, kv = torch.randn(BATCH, LENGTH, D_MODEL), torch.randn(BATCH, 7, D_MODEL)
    out, weights = ours(q, kv, kv)
    ref_out, ref_w = ref(q, kv, kv, average_attn_weights=False)
    assert out.shape == (BATCH, LENGTH, D_MODEL) and weights.shape == (BATCH, HEADS, LENGTH, 7)
    assert_close(out, ref_out, atol=1e-5)
    assert_close(weights, ref_w, atol=1e-5)


def test_mha_invalid_heads() -> None:
    with pytest.raises(ValueError):
        a.MultiHeadAttention(10, 3)


def test_sinusoidal_positional_encoding() -> None:
    pe = a.sinusoidal_positional_encoding(50, 8)
    assert pe.shape == (50, 8)
    assert_close(pe[0], torch.tensor([0.0, 1.0] * 4))
    assert pe[3, 0].item() == pytest.approx(math.sin(3))
    assert pe[3, 3].item() == pytest.approx(math.cos(3 / 10000 ** (2 / 8)), abs=1e-6)
    assert pe.abs().max() <= 1.0


@pytest.mark.parametrize("norm_first", [True, False])
def test_encoder_block(norm_first: bool) -> None:
    block = a.TransformerEncoderBlock(D_MODEL, HEADS, norm_first)
    x = torch.randn(BATCH, LENGTH, D_MODEL)
    out = block(x)
    assert out.shape == x.shape, "Forma incorrecta."
    assert block.ffn[0].out_features == 4 * D_MODEL, "FFN de 4·d."
    h = block.norm1(x) if norm_first else x
    attn = block.attn(h, h, h)[0]
    if norm_first:
        mid = x + attn
        expected = mid + block.ffn(block.norm2(mid))
    else:
        mid = block.norm1(x + attn)
        expected = block.norm2(mid + block.ffn(mid))
    assert_close(out, expected, atol=1e-5)


def test_encoder_block_causal() -> None:
    """Con máscara causal, cambiar el último token no afecta a los anteriores."""

    block = a.TransformerEncoderBlock(D_MODEL, HEADS)
    x = torch.randn(1, LENGTH, D_MODEL)
    y = x.clone()
    y[:, -1] += 5
    mask = a.causal_mask(LENGTH)
    assert_close(block(x, mask)[:, :-1], block(y, mask)[:, :-1], atol=1e-5)


def test_hard_distillation_loss() -> None:
    student = torch.randn(4, 3, requires_grad=True)
    teacher = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0])
    loss = a.hard_distillation_loss(student, teacher, targets)
    expected = 0.5 * F.cross_entropy(student, targets) + 0.5 * F.cross_entropy(student, teacher.argmax(1))
    assert_close(loss, expected)
    loss.backward()
    assert teacher.grad is None, "Sin gradiente hacia el profesor."
    same = a.hard_distillation_loss(student, F.one_hot(targets, 3).float(), targets)
    assert_close(same, F.cross_entropy(student, targets))


# --------------------------------------------------------------------------- #
# ViT                                                                         #
# --------------------------------------------------------------------------- #


def test_patch_embedding() -> None:
    embed = vit.PatchEmbedding(image_size=8, patch_size=4, in_channels=3, d_model=D_MODEL)
    x = torch.randn(BATCH, 3, 8, 8)
    out = embed(x)
    assert embed.num_patches == 4
    assert out.shape == (BATCH, 4, D_MODEL), "Forma incorrecta."
    # parche 1 = fila 0, columna 1 → píxeles [0:4, 4:8]
    expected = embed.proj(x[:, :, 0:4, 4:8]).flatten(1)
    assert_close(out[:, 1], expected, atol=1e-5)
    with pytest.raises(ValueError):
        vit.PatchEmbedding(10, 4, 3, D_MODEL)


def test_vit_paper_shape() -> None:
    """ViT-B/16 sobre 224×224: 196 parches + CLS = 197 tokens."""

    model = vit.VisionTransformer(224, 16, num_classes=10, d_model=32, depth=1, num_heads=4)
    assert model.pos_embedding.shape == (1, 197, 32)
    assert model.cls_token.shape == (1, 1, 32)


def test_vit_forward_and_tokens() -> None:
    model = vit.VisionTransformer(16, 4, num_classes=5, d_model=D_MODEL, depth=2, num_heads=HEADS)
    x = torch.randn(BATCH, 3, 16, 16)
    tokens = model.tokens(x)
    assert tokens.shape == (BATCH, 17, D_MODEL)
    assert_close(model(x), model.head(tokens[:, 0]), atol=1e-5)
    assert len(model.blocks) == 2 and all(b.norm_first for b in model.blocks)


def test_vit_cls_token_is_used() -> None:
    model = vit.VisionTransformer(8, 4, num_classes=3, d_model=D_MODEL, depth=1, num_heads=HEADS)
    x = torch.randn(1, 3, 8, 8)
    before = model(x)
    with torch.no_grad():
        # (sumar una constante no serviría: la LayerNorm la anula)
        model.cls_token.add_(torch.randn_like(model.cls_token))
    assert not torch.allclose(model(x), before), "La salida depende del token CLS."


def test_vit_trains() -> None:
    model = vit.VisionTransformer(8, 4, num_classes=2, d_model=D_MODEL, depth=1, num_heads=HEADS)
    x, y = torch.randn(8, 3, 8, 8), torch.randint(0, 2, (8,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(30):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first, "La pérdida baja (sobreajusta un lote pequeño)."
