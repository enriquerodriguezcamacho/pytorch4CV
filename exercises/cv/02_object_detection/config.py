"""
CV02 · Detección de objetos: configuración de YOLOv1 — ejercicios.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class YOLOConfig:
    """
    Hiperparámetros de YOLOv1.

    Disposición del vector de cada celda (como en la práctica 2), de longitud
    ``C + 5B``: ``[p_1, ..., p_C, conf_1, x_1, y_1, w_1, h_1, conf_2, x_2, ...]``.
    ``x, y`` son relativas a la celda (en [0, 1]) y ``w, h`` relativas a la imagen.

    Attributes:
        S: tamaño de la rejilla (S × S celdas).
        B: cajas predichas por celda.
        C: número de clases.
        lambda_coord: peso de la pérdida de coordenadas.
        lambda_noobj: peso de la confianza en celdas sin objeto.
        eps: constante numérica (raíces y divisiones).
    """

    S: int = 7
    B: int = 2
    C: int = 20
    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5
    eps: float = 1e-6

    @property
    def cell_size(self) -> int:
        """Longitud del vector de cada celda: ``5B + C``."""

        return 5 * self.B + self.C
