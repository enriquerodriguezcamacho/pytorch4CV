"""
F11 · Arquitecturas: modelos de secuencias — soluciones.

Una celda LSTM desde cero, un clasificador de secuencias con ``nn.LSTM`` (con
*packed sequences*) y un clasificador con ``nn.TransformerEncoder``.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def lstm_cell_step(
    x: torch.Tensor,
    h: torch.Tensor,
    c: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Un paso de una celda LSTM, con los pesos en el formato de ``nn.LSTMCell``.

    Las puertas se apilan en el orden de PyTorch ``(i, f, g, o)``:
    ``i, f, o = σ(·)``, ``g = tanh(·)``, ``c' = f ⊙ c + i ⊙ g``, ``h' = o ⊙ tanh(c')``.

    Args:
        x: entrada en el instante t. Dimensions: [batch, input_size].
        h: estado oculto anterior. Dimensions: [batch, hidden_size].
        c: estado de celda anterior. Dimensions: [batch, hidden_size].
        weight_ih: pesos entrada→puertas. Dimensions: [4 * hidden_size, input_size].
        weight_hh: pesos oculto→puertas. Dimensions: [4 * hidden_size, hidden_size].
        bias_ih: sesgo. Dimensions: [4 * hidden_size].
        bias_hh: sesgo. Dimensions: [4 * hidden_size].

    Returns:
        ``(h', c')``. Dimensions: [batch, hidden_size] cada uno.
    """

    # BEGIN SOLUTION
    gates = x @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh
    i, f, g, o = gates.chunk(4, dim=1)
    i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
    c_new = f * c + i * g
    h_new = o * torch.tanh(c_new)
    return h_new, c_new
    # END SOLUTION


class LSTMClassifier(nn.Module):
    """
    Clasificador de secuencias: ``nn.LSTM(batch_first=True)`` + ``nn.Linear`` sobre el
    último estado oculto de la última capa.

    Atributos que hay que crear:
        * ``self.lstm``: ``nn.LSTM(input_size, hidden_size, num_layers,
          batch_first=True, bidirectional=bidirectional)``.
        * ``self.fc``: ``nn.Linear(hidden_size * num_directions, num_classes)``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        """
        Constructor.

        Args:
            input_size: características por instante (p.ej. dimensión de un embedding
                de frame de vídeo o de una columna de una imagen de texto).
            hidden_size: tamaño del estado oculto.
            num_classes: número de clases.
            num_layers: número de capas LSTM apiladas.
            bidirectional: si la LSTM es bidireccional.
        """

        super().__init__()
        self.bidirectional = bidirectional

        # BEGIN SOLUTION
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * directions, num_classes)
        # END SOLUTION

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.

        Si se dan ``lengths``, la entrada se empaqueta con ``pack_padded_sequence``
        (``enforce_sorted=False``) para que el relleno no afecte al último estado.
        Si la LSTM es bidireccional se concatenan los últimos estados de las dos
        direcciones de la última capa (``h_n[-2]`` y ``h_n[-1]``).

        Args:
            x: secuencias (rellenas con ceros al final). Dimensions:
                [batch, seq_len, input_size].
            lengths: longitud real de cada secuencia (en CPU). Dimensions: [batch].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # BEGIN SOLUTION
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(x)  # h_n: [num_layers * dirs, B, hidden]
        last = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        return self.fc(last)
        # END SOLUTION


class TransformerSequenceClassifier(nn.Module):
    """
    Clasificador de secuencias con un encoder Transformer y *mean pooling*.

    Atributos que hay que crear:
        * ``self.proj``: ``nn.Linear(input_size, d_model)``.
        * ``self.pos_embedding``: ``nn.Parameter`` de ceros ``[1, max_len, d_model]``.
        * ``self.encoder``: ``nn.TransformerEncoder`` con ``num_layers`` capas
          ``nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model,
          dropout=dropout, batch_first=True)``.
        * ``self.fc``: ``nn.Linear(d_model, num_classes)``.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 128,
        dropout: float = 0.0,
    ) -> None:
        """
        Constructor.

        Args:
            input_size: características por instante.
            num_classes: número de clases.
            d_model: dimensión del modelo.
            nhead: número de cabezas de atención.
            num_layers: número de capas del encoder.
            max_len: longitud máxima de secuencia (embedding posicional aprendido).
            dropout: dropout de las capas del encoder.
        """

        super().__init__()

        # BEGIN SOLUTION
        self.proj = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.fc = nn.Linear(d_model, num_classes)
        # END SOLUTION

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: secuencias. Dimensions: [batch, seq_len, input_size].
            padding_mask: ``True`` en las posiciones de relleno (se ignoran en la
                atención y en la media). Dimensions: [batch, seq_len].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # BEGIN SOLUTION
        t = x.shape[1]
        z = self.proj(x) + self.pos_embedding[:, :t]
        z = self.encoder(z, src_key_padding_mask=padding_mask)
        if padding_mask is None:
            pooled = z.mean(dim=1)
        else:
            keep = (~padding_mask).unsqueeze(-1).float()  # [B, T, 1]
            pooled = (z * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        return self.fc(pooled)
        # END SOLUTION
