"""
Tests de CV09 · Audio: características, espectrogramas, SpecAugment, CTC y WER.
"""

import math

import pytest
import torch
from helpers import assert_close, load

f = load("cv/09_audio/features.py")
s = load("cv/09_audio/speech.py")

SAMPLE_RATE = 16000


def _sine(freq: float, seconds: float = 0.1, sr: int = SAMPLE_RATE) -> torch.Tensor:
    t = torch.arange(int(seconds * sr)) / sr
    return torch.sin(2 * math.pi * freq * t)


# --------------------------------------------------------------------------- #
# Señal                                                                       #
# --------------------------------------------------------------------------- #


def test_frame_signal() -> None:
    x = torch.arange(10.0)
    frames = f.frame_signal(x, 4, 3)
    assert frames.shape == (3, 4), "1 + (10 − 4) // 3 = 3 frames."
    assert frames[1].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert f.frame_signal(torch.randn(2, 5, 100), 10, 5).shape == (2, 5, 19, 10)


def test_zero_crossing_rate() -> None:
    alternating = torch.tensor([1.0, -1.0] * 4)
    assert_close(f.zero_crossing_rate(alternating, 8, 8), torch.tensor([1.0]))
    assert_close(f.zero_crossing_rate(torch.ones(8), 4, 4), torch.zeros(2))
    low, high = _sine(100), _sine(4000)
    assert f.zero_crossing_rate(high, 400, 200).mean() > f.zero_crossing_rate(low, 400, 200).mean()


def test_rms_energy() -> None:
    x = torch.cat([torch.zeros(4), torch.full((4,), 3.0)])
    assert_close(f.rms_energy(x, 4, 4), torch.tensor([0.0, 3.0]))
    sine = _sine(500, seconds=0.2)
    assert_close(f.rms_energy(sine, 3200, 3200), torch.full((1,), 1 / math.sqrt(2)), atol=1e-3)


def test_mel_conversions() -> None:
    assert f.hz_to_mel(torch.tensor(700.0)).item() == pytest.approx(2595 * math.log10(2))
    assert f.hz_to_mel(torch.tensor(0.0)).item() == 0.0
    freqs = torch.tensor([0.0, 440.0, 1000.0, 8000.0])
    assert_close(f.mel_to_hz(f.hz_to_mel(freqs)), freqs, atol=1e-2)


def test_mel_filterbank_properties() -> None:
    fb = f.mel_filterbank(n_freqs=201, f_min=0.0, f_max=8000.0, n_mels=10, sample_rate=SAMPLE_RATE)
    assert fb.shape == (201, 10), "Forma incorrecta: [n_freqs, n_mels]."
    assert (fb >= 0).all() and fb.max() <= 1 + 1e-6, "Triángulos con pico 1."
    peaks = fb.argmax(dim=0)
    assert (peaks[1:] > peaks[:-1]).all(), "Los filtros van de graves a agudos."
    widths = (fb > 0).sum(0)
    assert widths[-1] > widths[0], "En Hz los filtros agudos son más anchos."


def test_mel_filterbank_matches_torchaudio() -> None:
    taf = pytest.importorskip("torchaudio.functional")
    fb = f.mel_filterbank(257, 20.0, 7600.0, 40, SAMPLE_RATE)
    ref = taf.melscale_fbanks(257, 20.0, 7600.0, 40, SAMPLE_RATE, norm=None, mel_scale="htk")
    assert_close(fb, ref, atol=1e-5)


# --------------------------------------------------------------------------- #
# Espectro                                                                    #
# --------------------------------------------------------------------------- #


def test_dft_matches_fft() -> None:
    x = torch.randn(32)
    out = f.dft(x)
    assert out.is_complex() and out.shape == (32,)
    assert_close(out, torch.fft.fft(x), atol=1e-4)


def test_dft_pure_tone() -> None:
    """Un coseno de k₀ = 3 ciclos en N = 16 muestras: picos N/2 en k = 3 y k = 13."""

    n = torch.arange(16)
    spectrum = f.dft(torch.cos(2 * math.pi * 3 * n / 16)).abs()
    assert spectrum[3].item() == pytest.approx(8.0, abs=1e-4)
    assert spectrum[13].item() == pytest.approx(8.0, abs=1e-4)
    assert spectrum[[0, 1, 2, 4, 8]].max() < 1e-4


def test_stft_magnitude_matches_torch() -> None:
    x = torch.randn(2, 1000)
    out = f.stft_magnitude(x, n_fft=256, hop_length=64)
    ref = torch.stft(x, 256, 64, window=torch.hann_window(256), center=False, return_complex=True).abs()
    assert out.shape == ref.shape == (2, 129, 1 + (1000 - 256) // 64)
    assert_close(out, ref, atol=1e-4)


def test_amplitude_to_db() -> None:
    spec = torch.tensor([1.0, 10.0, 100.0, 0.0])
    assert_close(f.amplitude_to_db(spec), torch.tensor([0.0, 10.0, 20.0, -100.0]))
    assert_close(f.amplitude_to_db(spec[:3], multiplier=20.0), torch.tensor([0.0, 20.0, 40.0]))
    assert_close(f.amplitude_to_db(spec, top_db=15.0), torch.tensor([5.0, 10.0, 20.0, 5.0]))


def test_spectral_centroid() -> None:
    freqs = torch.tensor([0.0, 100.0, 200.0])
    spec = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])  # [freq=3, frames=2]
    assert_close(f.spectral_centroid(spec, freqs), torch.tensor([100.0, 100.0]))
    spec = torch.tensor([[0.0], [0.0], [2.0]])
    assert_close(f.spectral_centroid(spec, freqs), torch.tensor([200.0]))
    bright = f.stft_magnitude(_sine(4000), 512, 256)
    dark = f.stft_magnitude(_sine(300), 512, 256)
    bins = torch.linspace(0, SAMPLE_RATE / 2, 257)
    assert f.spectral_centroid(bright, bins).mean() > f.spectral_centroid(dark, bins).mean()


# --------------------------------------------------------------------------- #
# SpecAugment                                                                 #
# --------------------------------------------------------------------------- #


def test_mask_along_axis() -> None:
    spec = torch.ones(1, 6, 8)
    freq = f.mask_along_axis(spec, 2, 3, axis=-2)
    assert (freq[0, 2:5] == 0).all() and freq[0, [0, 1, 5]].eq(1).all()
    time = f.mask_along_axis(spec, 6, 5, axis=-1)
    assert (time[..., 6:] == 0).all() and time[..., :6].eq(1).all(), "Se recorta al borde."
    assert spec.eq(1).all(), "Trabaja sobre una copia."
    with pytest.raises(ValueError):
        f.mask_along_axis(spec, 0, 1, axis=0)


def test_spec_augment_only_in_training() -> None:
    spec = torch.rand(2, 10, 20) + 0.1
    assert torch.equal(f.spec_augment(spec, 4, 6, training=False), spec)
    torch.manual_seed(0)
    out = f.spec_augment(spec, 4, 6)
    torch.manual_seed(0)
    fw = int(torch.randint(0, 5, (1,)))
    fs = int(torch.randint(0, 10 - fw + 1, (1,)))
    tw = int(torch.randint(0, 7, (1,)))
    ts = int(torch.randint(0, 20 - tw + 1, (1,)))
    expected = f.mask_along_axis(f.mask_along_axis(spec, fs, fw, -2), ts, tw, -1)
    assert torch.equal(out, expected), "Frecuencia primero, luego tiempo; width antes que start."


# --------------------------------------------------------------------------- #
# Modelos y CTC                                                               #
# --------------------------------------------------------------------------- #


def test_audio_cnn_shapes() -> None:
    model = s.AudioCNN(num_classes=10, channels=4)
    assert model(torch.randn(3, 40, 50)).shape == (3, 10)
    assert model(torch.randn(3, 1, 40, 50)).shape == (3, 10)


def test_speech_crnn_output_format() -> None:
    model = s.SpeechCRNN(n_mels=20, num_classes=6, channels=2, hidden=8)
    out = model(torch.randn(3, 20, 30))
    assert out.shape == (30, 3, 6), "Formato de CTCLoss: [T, B, C]."
    assert_close(out.exp().sum(-1), torch.ones(30, 3), atol=1e-5)
    assert model.gru.bidirectional


def test_ctc_training_step_decreases_loss() -> None:
    model = s.SpeechCRNN(n_mels=8, num_classes=5, channels=2, hidden=16)
    spec = torch.randn(2, 8, 20)
    targets = torch.tensor([[1, 2, 3], [4, 4, 0]])
    input_lengths = torch.tensor([20, 20])
    target_lengths = torch.tensor([3, 2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = s.ctc_loss(model(spec), targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_ctc_greedy_decode() -> None:
    def one_hot_path(path: list[int], num_classes: int = 4) -> torch.Tensor:
        return torch.nn.functional.one_hot(torch.tensor(path), num_classes).float().log_softmax(-1)

    assert s.ctc_greedy_decode(one_hot_path([1, 1, 0, 1, 2, 2])) == [1, 1, 2], "a a - a b b → a a b."
    assert s.ctc_greedy_decode(one_hot_path([0, 0, 0])) == []
    assert s.ctc_greedy_decode(one_hot_path([3, 0, 3, 3, 0])) == [3, 3]
    assert s.ctc_greedy_decode(one_hot_path([2, 1, 1]), blank=1) == [2]


def test_levenshtein() -> None:
    assert s.levenshtein(list("kitten"), list("sitting")) == 3
    assert s.levenshtein([], ["a", "b"]) == 2
    assert s.levenshtein(["a"], []) == 1
    assert s.levenshtein(["x", "y"], ["x", "y"]) == 0


def test_word_error_rate() -> None:
    assert s.word_error_rate("el gato come pescado frito", "el perro come pescado crudo") == pytest.approx(0.4)
    assert s.word_error_rate("hola mundo", "hola mundo") == 0.0
    assert s.word_error_rate("hola", "hola que tal") == pytest.approx(2.0), "Inserciones: puede superar 1."
    with pytest.raises(ValueError):
        s.word_error_rate("", "algo")
