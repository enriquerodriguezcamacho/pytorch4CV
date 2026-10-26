"""
Tests de F11 · Arquitecturas: LSTM y Transformer para secuencias.
"""

import pytest
import torch
from helpers import assert_close, load
from torch import nn

m = load("fundamentals/11_architectures/sequence_models.py")

BATCH = 3
SEQ_LEN = 6
INPUT_SIZE = 5
HIDDEN = 8
NUM_CLASSES = 4


# ---------------------------------------------------------------- lstm_cell_step


def test_lstm_cell_step_vs_pytorch() -> None:
    """Coincide con ``nn.LSTMCell`` usando sus propios pesos."""

    cell = nn.LSTMCell(INPUT_SIZE, HIDDEN)
    x = torch.randn(BATCH, INPUT_SIZE)
    h, c = torch.randn(BATCH, HIDDEN), torch.randn(BATCH, HIDDEN)
    h_ref, c_ref = cell(x, (h, c))
    h_new, c_new = m.lstm_cell_step(
        x, h, c, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh
    )
    assert h_new.shape == (BATCH, HIDDEN) and c_new.shape == (BATCH, HIDDEN), "Formas."
    assert_close(h_new, h_ref)
    assert_close(c_new, c_ref)


def test_lstm_cell_step_hand_value() -> None:
    """Pesos a cero: todas las puertas valen σ(0)=0.5 y g=tanh(0)=0 → c'=c/2."""

    hidden = 2
    zeros_ih, zeros_hh = torch.zeros(4 * hidden, 3), torch.zeros(4 * hidden, hidden)
    zb = torch.zeros(4 * hidden)
    c = torch.tensor([[2.0, -4.0]])
    h_new, c_new = m.lstm_cell_step(torch.ones(1, 3), torch.ones(1, hidden), c, zeros_ih, zeros_hh, zb, zb)
    assert torch.isclose(c_new, torch.tensor([[1.0, -2.0]])).all(), "c' = f·c + i·g."
    assert torch.isclose(h_new, 0.5 * torch.tanh(torch.tensor([[1.0, -2.0]]))).all(), (
        "h' = o·tanh(c')."
    )


def test_lstm_cell_unrolled_vs_nn_lstm() -> None:
    """Desenrollar la celda en el tiempo reproduce ``nn.LSTM``."""

    lstm = nn.LSTM(INPUT_SIZE, HIDDEN, batch_first=True)
    x = torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)
    out_ref, (h_ref, _) = lstm(x)
    h = c = torch.zeros(BATCH, HIDDEN)
    for t in range(SEQ_LEN):
        h, c = m.lstm_cell_step(
            x[:, t], h, c, lstm.weight_ih_l0, lstm.weight_hh_l0, lstm.bias_ih_l0, lstm.bias_hh_l0
        )
    assert_close(h, h_ref[0])
    assert_close(h, out_ref[:, -1])


# ---------------------------------------------------------------- LSTMClassifier


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_lstm_classifier_shape(num_layers: int, bidirectional: bool) -> None:
    """Formas de salida y configuración de la LSTM."""

    model = m.LSTMClassifier(INPUT_SIZE, HIDDEN, NUM_CLASSES, num_layers, bidirectional)
    out = model(torch.randn(BATCH, SEQ_LEN, INPUT_SIZE))
    assert out.shape == (BATCH, NUM_CLASSES), "Forma incorrecta."
    assert model.lstm.batch_first, "Usa batch_first=True."
    assert model.fc.in_features == HIDDEN * (2 if bidirectional else 1), "Entrada de fc."


def test_lstm_classifier_uses_last_hidden_state() -> None:
    """La salida es ``fc(h_n[-1])`` de la última capa."""

    model = m.LSTMClassifier(INPUT_SIZE, HIDDEN, NUM_CLASSES, num_layers=2)
    x = torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)
    out_seq, (h_n, _) = model.lstm(x)
    assert_close(model(x), model.fc(h_n[-1]))
    assert_close(model(x), model.fc(out_seq[:, -1]))


@pytest.mark.parametrize("bidirectional", [False, True])
def test_lstm_classifier_packed_ignores_padding(bidirectional: bool) -> None:
    """Con ``lengths`` el relleno no cambia la salida de cada secuencia."""

    model = m.LSTMClassifier(INPUT_SIZE, HIDDEN, NUM_CLASSES, bidirectional=bidirectional)
    lengths = torch.tensor([SEQ_LEN, 2, 4])
    x = torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)
    for i, n in enumerate(lengths):
        x[i, n:] = 0.0
    out = model(x, lengths)
    for i, n in enumerate(lengths):
        alone = model(x[i : i + 1, :n])
        assert_close(out[i : i + 1], alone)
    # sin lengths, el relleno sí cambia la secuencia corta
    assert not torch.allclose(model(x)[1], out[1]), "Sin empaquetar debería cambiar."


def test_lstm_classifier_gradients() -> None:
    """Todos los parámetros reciben gradiente."""

    model = m.LSTMClassifier(INPUT_SIZE, HIDDEN, NUM_CLASSES)
    loss = nn.functional.cross_entropy(
        model(torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)), torch.tensor([0, 1, 2])
    )
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"Sin gradiente en {name}."


# ---------------------------------------------------------------- Transformer


def test_transformer_classifier_shape() -> None:
    """Forma de salida y estructura del encoder."""

    model = m.TransformerSequenceClassifier(INPUT_SIZE, NUM_CLASSES, d_model=16, nhead=2)
    out = model(torch.randn(BATCH, SEQ_LEN, INPUT_SIZE))
    assert out.shape == (BATCH, NUM_CLASSES), "Forma incorrecta."
    assert len(model.encoder.layers) == 2, "Número de capas del encoder."
    assert model.encoder.layers[0].self_attn.batch_first, "Usa batch_first=True."
    assert model.pos_embedding.shape == (1, 128, 16), "Forma del embedding posicional."


def test_transformer_classifier_padding_mask() -> None:
    """Con máscara de relleno, el contenido del relleno no influye."""

    model = m.TransformerSequenceClassifier(INPUT_SIZE, NUM_CLASSES, d_model=16, nhead=2).eval()
    x = torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)
    mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    mask[0, 3:] = True
    x2 = x.clone()
    x2[0, 3:] = 100.0 * torch.randn(SEQ_LEN - 3, INPUT_SIZE)
    assert_close(model(x, mask), model(x2, mask), atol=1e-4)
    assert_close(model(x, mask)[:1], model(x[:1, :3]), atol=1e-4)
