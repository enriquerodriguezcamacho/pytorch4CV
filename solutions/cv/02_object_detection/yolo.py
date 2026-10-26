"""
CV02 · Detección de objetos: YOLOv1 (codificación, pérdida y decodificación) — soluciones.
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

    # BEGIN SOLUTION
    target = torch.zeros(S, S, 5 * B + C)
    for (cx, cy, w, h), label in zip(boxes.tolist(), labels.tolist()):
        i, j = min(int(cy * S), S - 1), min(int(cx * S), S - 1)
        if target[i, j, C] == 1:
            continue
        target[i, j, int(label)] = 1.0
        target[i, j, C] = 1.0
        target[i, j, C + 1 : C + 5] = torch.tensor([cx * S - j, cy * S - i, w, h])
    return target
    # END SOLUTION


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

        # BEGIN SOLUTION
        s, c = self.config.S, self.config.C
        scale = torch.tensor([1 / s, 1 / s, 1.0, 1.0], device=predictions.device)
        pred_boxes = self._split_boxes(predictions)[..., 1:5] * scale  # [b,S,S,B,4]
        gt_box = (targets[..., c + 1 : c + 5] * scale).unsqueeze(-2)  # [b,S,S,1,4]
        ious = iou_midpoint(pred_boxes, gt_box.expand_as(pred_boxes))  # [b,S,S,B]
        return ious.argmax(dim=-1)
        # END SOLUTION

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

        # BEGIN SOLUTION
        c, eps = self.config.C, self.config.eps
        boxes = self._split_boxes(predictions)  # [b, S, S, B, 5]
        index = best_box[..., None, None].expand(*best_box.shape, 1, 5)
        resp = boxes.gather(-2, index).squeeze(-2)[..., 1:5]  # [b, S, S, 4]
        tgt = targets[..., c + 1 : c + 5]
        pred_wh = torch.sign(resp[..., 2:4]) * torch.sqrt(resp[..., 2:4].abs() + eps)
        tgt_wh = torch.sqrt(tgt[..., 2:4] + eps)
        err = (resp[..., :2] - tgt[..., :2]).pow(2).sum(-1)
        err = err + (pred_wh - tgt_wh).pow(2).sum(-1)
        return self.config.lambda_coord * (obj_mask * err).sum() / predictions.shape[0]
        # END SOLUTION

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

        # BEGIN SOLUTION
        c = self.config.C
        confs = self._split_boxes(predictions)[..., 0]  # [b, S, S, B]
        resp_conf = confs.gather(-1, best_box.unsqueeze(-1)).squeeze(-1)
        err = (resp_conf - targets[..., c]).pow(2)
        return (obj_mask * err).sum() / predictions.shape[0]
        # END SOLUTION

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

        # BEGIN SOLUTION
        c = self.config.C
        confs = self._split_boxes(predictions)[..., 0]  # [b, S, S, B]
        err = (confs - targets[..., c : c + 1]).pow(2).sum(-1)
        noobj = ((1 - obj_mask) * err).sum() / predictions.shape[0]
        return self.config.lambda_noobj * noobj
        # END SOLUTION

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

        # BEGIN SOLUTION
        c = self.config.C
        err = (predictions[..., :c] - targets[..., :c]).pow(2).sum(-1)
        return (obj_mask * err).sum() / predictions.shape[0]
        # END SOLUTION

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

        # BEGIN SOLUTION
        s = self.config.S
        predictions = predictions.reshape(-1, s, s, self.config.cell_size)
        obj_mask = targets[..., self.config.C]
        best_box = self._responsible_box(predictions, targets)
        coord = self._coord_loss(predictions, targets, obj_mask, best_box)
        obj = self._object_loss(predictions, targets, obj_mask, best_box)
        noobj = self._noobject_loss(predictions, targets, obj_mask)
        cls = self._class_loss(predictions, targets, obj_mask)
        return coord, obj, noobj, cls, coord + obj + noobj + cls
        # END SOLUTION


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

    # BEGIN SOLUTION
    predictions = predictions.reshape(-1, S, S, 5 * B + C)
    batch = predictions.shape[0]
    class_prob, label = predictions[..., :C].max(dim=-1)  # [b, S, S]
    boxes = predictions[..., C : C + 5 * B].unflatten(-1, (B, 5))  # [b, S, S, B, 5]

    rows = torch.arange(S, dtype=predictions.dtype).view(1, S, 1, 1)
    cols = torch.arange(S, dtype=predictions.dtype).view(1, 1, S, 1)
    cx = (cols + boxes[..., 1]) / S * img_w
    cy = (rows + boxes[..., 2]) / S * img_h
    w, h = boxes[..., 3] * img_w, boxes[..., 4] * img_h
    score = boxes[..., 0] * class_prob.unsqueeze(-1)
    label = label.unsqueeze(-1).expand(-1, -1, -1, B).to(predictions.dtype)
    out = torch.stack(
        [label, score, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
    )
    return out.reshape(batch, S * S * B, 6)
    # END SOLUTION


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
        # BEGIN SOLUTION
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden, config.S * config.S * config.cell_size),
        )
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: mapa del *backbone*. Dimensions: [batch, channels, height, width].

        Returns:
            Predicción en rejilla. Dimensions: [batch, S, S, 5B + C].
        """

        # BEGIN SOLUTION
        s = self.config.S
        return self.fc(x).view(-1, s, s, self.config.cell_size)
        # END SOLUTION
