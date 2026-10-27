"""
Tests de CV04 · 3D y vídeo: modelos de vídeo e inflado I3D.
"""

import pytest
import torch
from helpers import assert_close, load
from torch import nn

v = load("cv/04_3d_video/video.py")

BATCH, T, H, W = 2, 8, 12, 12
NUM_CLASSES = 5


def _video(t: int = T) -> torch.Tensor:
    return torch.randn(BATCH, t, 3, H, W)


def test_single_frame_averages_frame_logits() -> None:
    model = v.SingleFrameModel(NUM_CLASSES)
    video = _video()
    out = model(video)
    assert out.shape == (BATCH, NUM_CLASSES), "Forma incorrecta."
    expected = torch.stack([model.cnn(video[:, i]) for i in range(T)], dim=1).mean(1)
    assert_close(out, expected, atol=1e-5)


def test_late_fusion_concatenates_in_order() -> None:
    model = v.LateFusionModel(T, NUM_CLASSES, feat_dim=4)
    video = _video()
    out = model(video)
    assert out.shape == (BATCH, NUM_CLASSES), "Forma incorrecta."
    feats = torch.cat([model.cnn(video[:, i]) for i in range(T)], dim=1)
    assert_close(out, model.classifier(feats), atol=1e-5)


def test_early_fusion_channel_stacking() -> None:
    model = v.EarlyFusionModel(T, NUM_CLASSES)
    video = _video()
    out = model(video)
    assert out.shape == (BATCH, NUM_CLASSES), "Forma incorrecta."
    first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    assert first_conv.in_channels == 3 * T, "La primera conv recibe 3T canales."
    stacked = torch.cat([video[:, i] for i in range(T)], dim=1)  # frame 0 → canales 0..2
    assert_close(out, model.net(stacked), atol=1e-5)


def test_c3d_shape_and_permute() -> None:
    model = v.C3DModel(NUM_CLASSES)
    video = _video()
    out = model(video)
    assert out.shape == (BATCH, NUM_CLASSES), "Forma incorrecta."
    assert_close(out, model.net(video.transpose(1, 2)), atol=1e-5)
    pool = next(m for m in model.modules() if isinstance(m, nn.MaxPool3d))
    assert pool.kernel_size in ((1, 2, 2), [1, 2, 2]), "El primer pooling no reduce el tiempo."


@pytest.mark.parametrize("t", [3, 8])
def test_cnn_lstm_variable_length(t: int) -> None:
    model = v.CNNLSTMModel(NUM_CLASSES, feat_dim=4, hidden=6)
    assert model(_video(t)).shape == (BATCH, NUM_CLASSES)


def test_cnn_lstm_uses_last_hidden_state() -> None:
    model = v.CNNLSTMModel(NUM_CLASSES, feat_dim=4, hidden=6)
    video = _video()
    feats = torch.stack([model.cnn(video[:, i]) for i in range(T)], dim=1)
    outputs, _ = model.lstm(feats)
    assert_close(model(video), model.fc(outputs[:, -1]), atol=1e-5)


def test_slowfast_pathways() -> None:
    model = v.SlowFastModel(NUM_CLASSES, alpha=4, slow_channels=16, beta_inv=8)
    video = _video()
    out = model(video)
    assert out.shape == (BATCH, NUM_CLASSES), "Forma incorrecta."
    assert model.fc.in_features == 16 + 2, "Concatena vía lenta (16) y rápida (16 // 8)."
    x = video.transpose(1, 2)
    slow = model.slow(x[:, :, ::4]).flatten(1)
    fast = model.fast(x).flatten(1)
    assert_close(out, model.fc(torch.cat([slow, fast], 1)), atol=1e-5)


def test_inflate_conv2d_boring_video() -> None:
    """Sobre un vídeo con el mismo frame repetido, la Conv3d inflada = Conv2d (frames interiores)."""

    conv2d = nn.Conv2d(3, 4, 3, stride=2, padding=1)
    conv3d = v.inflate_conv2d(conv2d, time_kernel=3)
    assert isinstance(conv3d, nn.Conv3d)
    assert conv3d.weight.shape == (4, 3, 3, 3, 3), "Forma de los pesos incorrecta."
    assert_close(conv3d.weight.sum(2), conv2d.weight.detach(), atol=1e-6)

    frame = torch.randn(1, 3, 8, 8)
    boring = frame.unsqueeze(2).repeat(1, 1, 5, 1, 1)  # [1, 3, T=5, 8, 8]
    out3d = conv3d(boring)
    out2d = conv2d(frame)
    assert out3d.shape == (1, 4, 5, 4, 4), "Mantiene T y aplica el stride espacial."
    for t in range(1, 4):
        assert_close(out3d[:, :, t], out2d, atol=1e-5)


def test_inflate_conv2d_without_bias() -> None:
    conv3d = v.inflate_conv2d(nn.Conv2d(2, 2, 1, bias=False), time_kernel=5)
    assert conv3d.bias is None
    assert conv3d.padding == (2, 0, 0)
