"""
CV02 · Detección de objetos: YOLOv1 (codificación, pérdida y decodificación) — ejercicios.
"""

from __future__ import annotations

import torch
from torch import nn

from .boxes import iou_midpoint
from .config import YOLOConfig


def encode_targets(
    boxes: torch.Tensor, labels: torch.Tensor, S: int, B: int, C: int
) -> torch.Tensor:
    """
    Codifica el *ground truth* de una imagen en el tensor objetivo S × S × (5B + C).

    La celda responsable es la que contiene el centro: ``i = floor(cy · S)``
    (fila), ``j = floor(cx · S)`` (columna), recortadas a ``S - 1``. En esa celda
    se escribe el *one-hot* de la clase, ``conf_1 = 1`` y la caja en el primer
    hueco: ``(cx · S - j, cy · S - i, w, h)``. Los demás huecos quedan a 0. Si dos
    objetos caen en la misma celda, se queda el primero (limitación de YOLOv1).

    Args:
        boxes: cajas ``cxcywh`` normalizadas a [0, 1]. Dimensions: [N, 4].
        labels: clases (``int64``). Dimensions: [N].
        S: tamaño de la rejilla.
        B: cajas por celda.
        C: número de clases.

    Returns:
        Tensor objetivo. Dimensions: [S, S, 5B + C].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class YOLOLoss(nn.Module):
    """
    Pérdida de YOLOv1 (suma de errores cuadráticos, dividida entre el tamaño de lote).

    ``L = L_coord + L_obj + L_noobj + L_class`` con

    * ``L_coord = λ_coord Σ 1^obj_ij [(x - x̂)² + (y - ŷ)² + (√w - √ŵ)² + (√h - √ĥ)²]``
    * ``L_obj = Σ 1^obj_ij (C - Ĉ)²``
    * ``L_noobj = λ_noobj Σ_i (1 - 1^obj_i) Σ_j (C - Ĉ_j)²`` (todas las cajas de las
      celdas sin objeto)
    * ``L_class = Σ_i 1^obj_i Σ_c (p_i(c) - p̂_i(c))²``

    ``1^obj_ij`` vale 1 solo para el predictor ``j`` responsable: el de mayor IoU
    con la caja real de la celda ``i``.
    """

    def __init__(self, config: YOLOConfig | None = None) -> None:
        """
        Constructor de la clase (ya dado).

        Args:
            config: hiperparámetros; por defecto ``YOLOConfig()``.
        """

        super().__init__()

        self.config = config if config is not None else YOLOConfig()

    def _split_boxes(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Separa las ``B`` cajas de cada celda (ya dado).

        Args:
            predictions: Dimensions: [batch, S, S, 5B + C].

        Returns:
            Cajas ``(conf, x, y, w, h)``. Dimensions: [batch, S, S, B, 5].
        """

        c, b = self.config.C, self.config.B
        return predictions[..., c : c + 5 * b].unflatten(-1, (b, 5))

    def _responsible_box(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Índice del predictor responsable de cada celda (mayor IoU con la caja real).

        Para que el IoU tenga sentido, ``x`` e ``y`` (relativas a la celda) se
        dividen entre ``S`` y así quedan en la misma unidad que ``w`` y ``h``
        (relativas a la imagen); el desplazamiento de la celda es común a ambas
        cajas y no cambia el IoU. Usa :func:`iou_midpoint`. En caso de empate gana
        el primer predictor.

        Args:
            predictions: Dimensions: [batch, S, S, 5B + C].
            targets: Dimensions: [batch, S, S, 5B + C].

        Returns:
            Índice en ``[0, B)`` (``int64``). Dimensions: [batch, S, S].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def _coord_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        obj_mask: torch.Tensor,
        best_box: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pérdida de coordenadas del predictor responsable.

        Para la raíz de la predicción se usa ``sign(w) · sqrt(|w| + eps)`` (la red
        puede predecir valores negativos) y para el objetivo ``sqrt(w + eps)``.

        Args:
            predictions: Dimensions: [batch, S, S, 5B + C].
            targets: Dimensions: [batch, S, S, 5B + C].
            obj_mask: 1 si la celda tiene objeto. Dimensions: [batch, S, S].
            best_box: índice del predictor responsable. Dimensions: [batch, S, S].

        Returns:
            ``λ_coord · Σ (...) / batch``. Escalar.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def _object_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        obj_mask: torch.Tensor,
        best_box: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pérdida de confianza del predictor responsable en celdas con objeto.

        El objetivo es la confianza de ``targets`` (``conf_1 = 1``); el paper usa el
        IoU entre la predicción y la caja real, variante que también es válida.

        Args:
            predictions: Dimensions: [batch, S, S, 5B + C].
            targets: Dimensions: [batch, S, S, 5B + C].
            obj_mask: Dimensions: [batch, S, S].
            best_box: Dimensions: [batch, S, S].

        Returns:
            ``Σ 1^obj (C - Ĉ)² / batch``. Escalar.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def _noobject_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, obj_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pérdida de confianza de **todas** las cajas de las celdas sin objeto.

        Args:
            predictions: Dimensions: [batch, S, S, 5B + C].
            targets: Dimensions: [batch, S, S, 5B + C].
            obj_mask: Dimensions: [batch, S, S].

        Returns:
            ``λ_noobj · Σ (1 - 1^obj) Σ_j (C - Ĉ_j)² / batch``. Escalar.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def _class_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, obj_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Error cuadrático de las probabilidades de clase en celdas con objeto
        (una sola distribución por celda, no por caja).

        Args:
            predictions: Dimensions: [batch, S, S, 5B + C].
            targets: Dimensions: [batch, S, S, 5B + C].
            obj_mask: Dimensions: [batch, S, S].

        Returns:
            ``Σ_i 1^obj_i Σ_c (p - p̂)² / batch``. Escalar.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            predictions: salida de la red, aplanada o no. Dimensions:
                [batch, S * S * (5B + C)] o [batch, S, S, 5B + C].
            targets: objetivos. Dimensions: [batch, S, S, 5B + C].

        Returns:
            ``(coord, obj, noobj, class, total)``, todos escalares.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


def cells_to_boxes(
    predictions: torch.Tensor, S: int, B: int, C: int, img_w: float, img_h: float
) -> torch.Tensor:
    """
    Convierte la salida S × S × (5B + C) en cajas en coordenadas de imagen.

    Para la caja ``b`` de la celda ``(i, j)`` (fila ``i``, columna ``j``):
    ``cx = (j + x) / S · img_w``, ``cy = (i + y) / S · img_h``, ``w = w · img_w``,
    ``h = h · img_h``. La clase es ``argmax_c p(c)`` y el *score* es la confianza
    específica de clase ``conf · max_c p(c)``.

    Args:
        predictions: Dimensions: [batch, S, S, 5B + C] (o aplanada
            [batch, S * S * (5B + C)]).
        S: tamaño de la rejilla.
        B: cajas por celda.
        C: número de clases.
        img_w: ancho de la imagen en píxeles.
        img_h: alto de la imagen en píxeles.

    Returns:
        Filas ``(label, score, x1, y1, x2, y2)``, ordenadas por celda (por filas) y
        luego por caja. Dimensions: [batch, S * S * B, 6].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class YOLOv1Head(nn.Module):
    """
    Cabeza de YOLOv1: dos capas FC sobre el mapa del *backbone* y *reshape* a la
    rejilla. Activación *leaky ReLU* (0.1) en la oculta y lineal a la salida.
    """

    def __init__(self, in_features: int, hidden: int, config: YOLOConfig) -> None:
        """
        Constructor de la clase.

        Args:
            in_features: tamaño del mapa del *backbone* aplanado (``C·H·W``).
            hidden: neuronas de la capa oculta (4096 en el paper).
            config: configuración (``S``, ``B``, ``C``).
        """

        super().__init__()

        self.config = config
        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: mapa del *backbone*. Dimensions: [batch, channels, height, width].

        Returns:
            Predicción en rejilla. Dimensions: [batch, S, S, 5B + C].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")
