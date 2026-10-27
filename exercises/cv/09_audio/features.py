"""
CV09 · Audio: características de la señal y espectrogramas — ejercicios.

Todo con ``torch`` (sin ``torchaudio``). Las señales son tensores ``[..., samples]``
y los espectrogramas ``[..., freq, frames]`` (la convención de ``torchaudio``).
"""

from __future__ import annotations

import math

import torch


def frame_signal(x: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    """
    Trocea la señal en *frames* solapados con ``Tensor.unfold`` (sin padding: el
    último trozo incompleto se descarta).

    Args:
        x: señal. Dimensions: [..., samples].
        frame_length: muestras por *frame*.
        hop_length: salto entre *frames*.

    Returns:
        Dimensions: [..., n_frames, frame_length] con
        ``n_frames = 1 + (samples - frame_length) // hop_length``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def zero_crossing_rate(x: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    """
    Tasa de cruces por cero por *frame*: fracción de las ``frame_length - 1``
    parejas consecutivas cuyo signo cambia (``x[n] · x[n+1] < 0``).

    Es alta en sonidos ruidosos o fricativos ("s") y baja en sonidos tonales.

    Args:
        x: señal. Dimensions: [..., samples].
        frame_length: muestras por *frame*.
        hop_length: salto entre *frames*.

    Returns:
        Valores en [0, 1]. Dimensions: [..., n_frames].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def rms_energy(x: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    """
    Energía RMS por *frame*: ``sqrt(mean(x²))`` (envolvente de la señal).

    Args:
        x: señal. Dimensions: [..., samples].
        frame_length: muestras por *frame*.
        hop_length: salto entre *frames*.

    Returns:
        Dimensions: [..., n_frames].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    """
    Escala mel (HTK): ``m = 2595 · log10(1 + f / 700)``.

    Args:
        freq: frecuencias en Hz. Dimensions: [*].

    Returns:
        Mels. Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    """
    Inversa de :func:`hz_to_mel`: ``f = 700 · (10^(m / 2595) - 1)``.

    Args:
        mel: mels. Dimensions: [*].

    Returns:
        Frecuencias en Hz. Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def mel_filterbank(
    n_freqs: int, f_min: float, f_max: float, n_mels: int, sample_rate: int
) -> torch.Tensor:
    """
    Banco de filtros triangulares en escala mel (igual que
    ``torchaudio.functional.melscale_fbanks(..., norm=None, mel_scale="htk")``).

    1. Frecuencias de los ``n_freqs`` *bins* del espectro:
       ``linspace(0, sample_rate // 2, n_freqs)``.
    2. ``n_mels + 2`` puntos equiespaciados **en mel** entre ``f_min`` y ``f_max``,
       pasados a Hz: ``f_pts``.
    3. El filtro ``m`` es un triángulo que sube de ``f_pts[m]`` a ``f_pts[m+1]`` y
       baja hasta ``f_pts[m+2]``:
       ``max(0, min((f - f_m) / (f_{m+1} - f_m), (f_{m+2} - f) / (f_{m+2} - f_{m+1})))``.

    Args:
        n_freqs: número de *bins* (``n_fft // 2 + 1``).
        f_min: frecuencia mínima.
        f_max: frecuencia máxima.
        n_mels: número de filtros.
        sample_rate: frecuencia de muestreo.

    Returns:
        Matriz para hacer ``mel = specᵀ @ fb``. Dimensions: [n_freqs, n_mels].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def dft(x: torch.Tensor) -> torch.Tensor:
    """
    DFT con la matriz de Fourier (``O(N²)``, igual que ``torch.fft.fft``):
    ``X_k = Σ_n x_n · exp(-2πi k n / N)``.

    Args:
        x: señal real. Dimensions: [N].

    Returns:
        Espectro complejo. Dimensions: [N].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def stft_magnitude(x: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    """
    Magnitud de la STFT sin *centering*: *frames* de ``n_fft`` muestras
    (:func:`frame_signal`), ventana de Hann periódica (``torch.hann_window``) y
    ``torch.fft.rfft`` de cada *frame*. Coincide con
    ``torch.stft(x, n_fft, hop_length, window=hann, center=False).abs()``.

    Args:
        x: señal. Dimensions: [..., samples].
        n_fft: tamaño de la FFT (= longitud del *frame*).
        hop_length: salto entre *frames*.

    Returns:
        Espectrograma de magnitud. Dimensions: [..., n_fft // 2 + 1, n_frames].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def amplitude_to_db(
    spec: torch.Tensor, multiplier: float = 10.0, amin: float = 1e-10, top_db: float | None = None
) -> torch.Tensor:
    """
    Paso a decibelios: ``multiplier · log10(max(spec, amin))`` (``10`` para
    potencia, ``20`` para amplitud). Con ``top_db`` se recorta todo lo que esté más
    de ``top_db`` dB por debajo del máximo (de todo el tensor).

    Args:
        spec: espectrograma (no negativo). Dimensions: [*].
        multiplier: 10 (potencia) o 20 (amplitud).
        amin: valor mínimo para evitar ``log(0)``.
        top_db: rango dinámico máximo.

    Returns:
        Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def spectral_centroid(spec: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Centroide espectral por *frame*: ``Σ_f f · S[f] / Σ_f S[f]`` (se suma ``1e-10``
    al denominador). Indica el "brillo" del sonido.

    Args:
        spec: espectrograma de magnitud. Dimensions: [..., freq, frames].
        freqs: frecuencia de cada *bin*. Dimensions: [freq].

    Returns:
        Dimensions: [..., frames].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def mask_along_axis(
    spec: torch.Tensor, start: int, width: int, axis: int, value: float = 0.0
) -> torch.Tensor:
    """
    Máscara de SpecAugment sobre una **copia**: pone a ``value`` las posiciones
    ``[start, start + width)`` del eje de frecuencia (``axis=-2``) o de tiempo
    (``axis=-1``).

    Args:
        spec: espectrograma. Dimensions: [..., freq, frames].
        start: primera posición enmascarada.
        width: número de posiciones enmascaradas.
        axis: ``-2`` (frecuencia) o ``-1`` (tiempo).
        value: valor de relleno.

    Returns:
        Espectrograma enmascarado. Dimensions: igual que ``spec``.

    Raises:
        ValueError: si ``axis`` no es ``-1`` ni ``-2``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def spec_augment(
    spec: torch.Tensor, freq_mask_param: int, time_mask_param: int, training: bool = True
) -> torch.Tensor:
    """
    SpecAugment (una máscara de frecuencia y una de tiempo), **solo en
    entrenamiento**. Para cada eje (primero frecuencia, luego tiempo):
    ``width = randint(0, param + 1)`` y ``start = randint(0, size - width + 1)``
    (``torch.randint``, en ese orden).

    Args:
        spec: espectrograma. Dimensions: [..., freq, frames].
        freq_mask_param: anchura máxima de la máscara de frecuencia.
        time_mask_param: anchura máxima de la máscara de tiempo.
        training: si ``False`` devuelve el espectrograma sin tocar.

    Returns:
        Dimensions: igual que ``spec``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
