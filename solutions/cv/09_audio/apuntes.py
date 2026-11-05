"""
CV09 · Ejercicios de código del tema 9 de los apuntes (sección 9.5) — soluciones.

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

    # BEGIN SOLUTION
    return round(milliseconds * sample_rate / 1000)
    # END SOLUTION


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

    # BEGIN SOLUTION
    output = []
    previous = None
    for symbol in path:
        if symbol != previous and symbol != blank:
            output.append(symbol)
        previous = symbol
    return "".join(output)
    # END SOLUTION


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

    # BEGIN SOLUTION
    ref, hyp = reference.split(), hypothesis.split()
    rows, cols = len(ref) + 1, len(hyp) + 1
    dist = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dist[i][0] = i
    for j in range(cols):
        dist[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = int(ref[i - 1] != hyp[j - 1])
            dist[i][j] = min(
                dist[i - 1][j - 1] + cost, dist[i - 1][j] + 1, dist[i][j - 1] + 1
            )

    i, j = len(ref), len(hyp)
    subs = dels = ins = 0
    while i > 0 or j > 0:
        cost = int(i > 0 and j > 0 and ref[i - 1] != hyp[j - 1])
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + cost:
            subs += cost
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return WERResult(dist[-1][-1] / len(ref), subs, dels, ins)
    # END SOLUTION
