"""
CV09 · Ejercicios de código del tema 9 de los apuntes (sección 9.5) — ejercicios.

El resto de ejercicios del tema es teórico: ``docs/SOLUCIONES_APUNTES.md``.
"""

from __future__ import annotations

from typing import NamedTuple


def duration_to_samples(milliseconds: float, sample_rate: int) -> int:
    """
    Ej. 2 · Número de muestras de una ventana o de un salto.

    Args:
        milliseconds: duración en milisegundos.
        sample_rate: frecuencia de muestreo en Hz.

    Returns:
        Muestras (entero más cercano).
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def ctc_collapse(path: list[str], blank: str = "∅") -> str:
    """
    Ej. 7 · Colapso de CTC: une repeticiones consecutivas y **después** quita los
    *blanks*. El *blank* permite letras dobles: ``l, ∅, l`` da ``ll``.

    Args:
        path: símbolo de cada instante.
        blank: símbolo *blank*.

    Returns:
        Texto colapsado.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class WERResult(NamedTuple):
    """Resultado de :func:`word_error_alignment`."""

    wer: float
    substitutions: int
    deletions: int
    insertions: int


def word_error_alignment(reference: str, hypothesis: str) -> WERResult:
    """
    Ej. 10 · WER con el desglose de ediciones de **una** alineación mínima.

    ``WER = (S + D + I) / N``, con ``N`` el número de palabras de la referencia. Puede
    haber varias alineaciones mínimas: todas dan el mismo WER. En caso de empate,
    al reconstruir desde el final se prefiere acierto o sustitución, después
    borrado y por último inserción.

    Args:
        reference: frase de referencia, con al menos una palabra.
        hypothesis: frase transcrita.

    Returns:
        WER y número de sustituciones, borrados e inserciones.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
