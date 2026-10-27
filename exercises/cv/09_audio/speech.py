"""
CV09 · Audio: modelos sobre espectrogramas, CTC y WER — ejercicios.
"""

from __future__ import annotations

import torch
from torch import nn


class AudioCNN(nn.Module):
    """
    Clasificador de sonidos sobre (log-)mel espectrogramas (estilo UrbanSound8K):
    el espectrograma se trata como una imagen de un canal.

    ``[Conv2d(1, c, 3, padding=1) → BatchNorm2d → ReLU → MaxPool2d(2)]`` y
    ``[Conv2d(c, 2c, 3, padding=1) → BatchNorm2d → ReLU]`` →
    ``AdaptiveAvgPool2d(1) → Flatten → Linear(2c, num_classes)``.
    """

    def __init__(self, num_classes: int, channels: int = 16) -> None:
        """
        Constructor de la clase.

        Args:
            num_classes: número de clases.
            channels: canales ``c`` del primer bloque.
        """

        super().__init__()

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            spec: espectrogramas. Dimensions: [batch, n_mels, frames] o
                [batch, 1, n_mels, frames].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class SpeechCRNN(nn.Module):
    """
    Modelo CNN + GRU para reconocimiento de voz con CTC (estilo Deep Speech 2).

    1. ``Conv2d(1, c, 3, padding=1) → ReLU`` sobre ``[B, 1, n_mels, T]`` (no reduce T).
    2. Se juntan canales y frecuencias: ``[B, T, c · n_mels]``.
    3. ``nn.GRU(c · n_mels, hidden, batch_first=True, bidirectional=True)``.
    4. ``Linear(2 · hidden, num_classes)`` y ``log_softmax`` sobre las clases.

    La salida se devuelve en el formato que espera ``nn.CTCLoss``: ``[T, B, C]``.
    La clase 0 es el *blank*.
    """

    def __init__(self, n_mels: int, num_classes: int, channels: int = 8, hidden: int = 32) -> None:
        """
        Constructor de la clase.

        Args:
            n_mels: bandas mel de la entrada.
            num_classes: símbolos incluyendo el *blank*.
            channels: canales ``c`` de la convolución.
            hidden: tamaño oculto de la GRU (por dirección).
        """

        super().__init__()

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            spec: espectrogramas. Dimensions: [batch, n_mels, T].

        Returns:
            Log-probabilidades. Dimensions: [T, batch, num_classes].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


def ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
) -> torch.Tensor:
    """
    Envoltorio de ``nn.CTCLoss(blank=blank, zero_infinity=True)`` (reducción media)
    que comprueba las formas. Ya dado.

    Args:
        log_probs: salida de :class:`SpeechCRNN`. Dimensions: [T, batch, C].
        targets: transcripciones concatenadas o con padding. Dimensions:
            [batch, S] o [sum(target_lengths)].
        input_lengths: frames válidos de cada muestra. Dimensions: [batch].
        target_lengths: longitud de cada transcripción. Dimensions: [batch].
        blank: índice del *blank*.

    Returns:
        Escalar.
    """

    assert log_probs.dim() == 3, "log_probs debe ser [T, B, C]."
    criterion = nn.CTCLoss(blank=blank, zero_infinity=True)
    return criterion(log_probs, targets, input_lengths, target_lengths)


def ctc_greedy_decode(log_probs: torch.Tensor, blank: int = 0) -> list[int]:
    """
    Decodificación *greedy* de CTC: ``argmax`` por *frame*, se **colapsan** las
    repeticiones consecutivas y después se **eliminan** los *blanks*
    (``a a - a b b → a a b``).

    Args:
        log_probs: log-probabilidades de una muestra. Dimensions: [T, C].
        blank: índice del *blank*.

    Returns:
        Secuencia decodificada (índices de símbolos).
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    """
    Distancia de edición (sustituciones + borrados + inserciones) con
    programación dinámica.

    Args:
        reference: secuencia de referencia.
        hypothesis: secuencia hipótesis.

    Returns:
        Número mínimo de ediciones.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    WER ``= (S + D + I) / N`` con ``N`` el número de palabras de la referencia
    (separadas por espacios con ``str.split``).

    Args:
        reference: transcripción correcta.
        hypothesis: transcripción del modelo.

    Returns:
        WER (puede superar 1 si hay muchas inserciones).

    Raises:
        ValueError: si la referencia no tiene palabras.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
