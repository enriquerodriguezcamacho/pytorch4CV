"""
F05 · Funciones de pérdida — ejercicios.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

Reduction = Literal["mean", "sum", "none"]


def reduce_loss(loss: torch.Tensor, reduction: Reduction = "mean") -> torch.Tensor:
    """
    Aplica la reducción indicada a una pérdida elemento a elemento.

    Args:
        loss: pérdida sin reducir. Dimensions: [*].
        reduction: ``"mean"`` (media), ``"sum"`` (suma) o ``"none"`` (sin reducir).

    Returns:
        Escalar si ``reduction`` es ``"mean"`` o ``"sum"``; el mismo tensor si es
        ``"none"``. Dimensions: [] o [*].

    Raises:
        ValueError: si ``reduction`` no es una de las tres opciones válidas.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Cross-entropy desde cero con la misma semántica que ``F.cross_entropy``.

    Con ``target`` de índices (``int64``) y log-probabilidades
    ``l = log_softmax(logits)``, la pérdida de una muestra es

    ``(1 - ε) · w[y] · (-l[y]) + (ε / C) · Σ_c w[c] · (-l[c])``

    y las muestras con ``y == ignore_index`` valen 0. Con ``reduction="mean"``
    se divide por ``Σ w[y_i]`` de las muestras no ignoradas (por el número de
    muestras válidas si no hay pesos). Con ``target`` de probabilidades
    (``float``, misma forma que ``logits``) la pérdida es
    ``-Σ_c w[c] · ỹ[c] · l[c]`` con ``ỹ = (1 - ε) · y + ε / C`` y la media se
    hace sobre el número de muestras.

    Args:
        logits: puntuaciones sin normalizar. Dimensions: [batch, classes] o
            [batch, classes, height, width].
        target: índices de clase ``int64``. Dimensions: [batch] o
            [batch, height, width]. O bien probabilidades ``float`` con la misma
            forma que ``logits``.
        weight: peso de cada clase o ``None``. Dimensions: [classes].
        ignore_index: valor del ``target`` que no contribuye (p.ej. 255 en VOC).
        label_smoothing: ``ε`` en ``[0, 1]``.
        reduction: ``"mean"``, ``"sum"`` o ``"none"``.

    Returns:
        Pérdida. Dimensions: [] o, con ``"none"``, [batch] / [batch, height, width].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    BCE sobre logits numéricamente estable (equivalente a ``BCEWithLogitsLoss``).

    ``ℓ = -[p_w · y · log σ(x) + (1 - y) · log(1 - σ(x))]``. Para que no haya
    overflow se usa ``log σ(x) = -softplus(-x)`` y
    ``softplus(z) = max(z, 0) + log(1 + exp(-|z|))``.

    Args:
        logits: logits ``x``. Dimensions: [*].
        target: etiquetas en ``[0, 1]`` (pueden ser suaves). Dimensions: [*].
        pos_weight: peso de los positivos ``p_w`` (uno por clase, broadcast sobre
            la última dimensión) o ``None``. Dimensions: [classes].
        reduction: ``"mean"``, ``"sum"`` o ``"none"``.

    Returns:
        Pérdida. Dimensions: [] o [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Smooth L1 (Fast R-CNN): cuadrática si ``|d| < β`` y lineal fuera.

    ``ℓ = 0.5 · d² / β`` si ``|d| < β``, ``|d| - 0.5 · β`` en otro caso. Con
    ``β = 0`` es exactamente L1.

    Args:
        pred: predicción (p.ej. desplazamientos de caja). Dimensions: [*].
        target: valor real. Dimensions: [*].
        beta: umbral ``β ≥ 0`` entre la zona cuadrática y la lineal.
        reduction: ``"mean"``, ``"sum"`` o ``"none"``.

    Returns:
        Pérdida. Dimensions: [] o [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Pérdida de Huber: ``0.5 · d²`` si ``|d| ≤ δ`` y ``δ · (|d| - 0.5 · δ)`` fuera.

    Cumple ``huber(δ) = δ · smooth_l1(β = δ)``.

    Args:
        pred: predicción. Dimensions: [*].
        target: valor real. Dimensions: [*].
        delta: umbral ``δ > 0``.
        reduction: ``"mean"``, ``"sum"`` o ``"none"``.

    Returns:
        Pérdida. Dimensions: [] o [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    Término suave de knowledge distillation (Hinton et al., 2015).

    ``T² · KL(p_T ‖ p_S)`` con ``p = softmax(z / T)`` y reducción
    ``batchmean`` (suma sobre clases, media sobre el lote). El factor ``T²``
    compensa que los gradientes escalan como ``1 / T²``.

    Args:
        student_logits: logits del estudiante. Dimensions: [batch, classes].
        teacher_logits: logits del profesor (no se propaga gradiente por ellos).
            Dimensions: [batch, classes].
        temperature: temperatura ``T > 0``.

    Returns:
        Pérdida escalar. Dimensions: [].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def sigmoid_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: Reduction = "none",
) -> torch.Tensor:
    """
    Focal loss binaria de RetinaNet (Lin et al., 2017).

    ``FL = -α_t · (1 - p_t)^γ · log(p_t)``, con ``p_t = p`` si ``y = 1`` y
    ``1 - p`` si ``y = 0``; ``α_t = α`` para positivos y ``1 - α`` para
    negativos. Si ``alpha < 0`` no se aplica ``α_t``. Misma semántica que
    ``torchvision.ops.sigmoid_focal_loss``.

    Args:
        logits: logits. Dimensions: [*].
        target: etiquetas binarias en float. Dimensions: [*].
        alpha: peso de los positivos (``< 0`` para desactivarlo).
        gamma: exponente de enfoque ``γ ≥ 0`` (``γ = 0`` → BCE ponderada).
        reduction: ``"mean"``, ``"sum"`` o ``"none"``.

    Returns:
        Pérdida. Dimensions: [] o [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """
    Soft Dice multiclase para segmentación (V-Net, Milletari et al., 2016).

    ``p = softmax(logits)`` y ``g = one_hot(target)``. Para cada clase ``c``,
    sumando sobre lote y píxeles válidos:
    ``Dice_c = (2 Σ p_c g_c + s) / (Σ p_c + Σ g_c + s)``. La pérdida es
    ``1 - mean_c Dice_c``. Los píxeles con ``target == ignore_index`` no cuentan.

    Args:
        logits: logits por píxel. Dimensions: [batch, classes, height, width].
        target: clase de cada píxel ``int64``. Dimensions: [batch, height, width].
        smooth: suavizado ``s`` (evita 0/0 y suaviza el gradiente).
        ignore_index: etiqueta a ignorar o ``None``.

    Returns:
        Pérdida escalar en ``[0, 1]``. Dimensions: [].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def generalized_iou_loss(
    boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """
    Pérdida GIoU (Rezatofighi et al., 2019) entre parejas de cajas.

    ``GIoU = IoU - |C \\ (A ∪ B)| / |C|``, con ``C`` la menor caja que contiene a
    ``A`` y ``B``. La pérdida es ``1 - GIoU ∈ [0, 2]``.

    Args:
        boxes1: cajas ``(x1, y1, x2, y2)``. Dimensions: [N, 4].
        boxes2: cajas emparejadas con ``boxes1``. Dimensions: [N, 4].
        eps: estabilidad numérica.

    Returns:
        Pérdida por pareja (sin reducir). Dimensions: [N].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
    p: float = 2.0,
    eps: float = 1e-6,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Triplet loss (FaceNet): ``max(0, d(a, p) - d(a, n) + margin)``.

    La distancia es la de ``F.pairwise_distance``: ``d(x, y) = ‖x - y + eps‖_p``.

    Args:
        anchor: embeddings ancla. Dimensions: [batch, dim].
        positive: embeddings de la misma identidad. Dimensions: [batch, dim].
        negative: embeddings de otra identidad. Dimensions: [batch, dim].
        margin: margen ``m > 0``.
        p: orden de la norma.
        eps: término de estabilidad dentro de la norma.
        reduction: ``"mean"``, ``"sum"`` o ``"none"``.

    Returns:
        Pérdida. Dimensions: [] o [batch].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def info_nce(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
    symmetric: bool = False,
) -> torch.Tensor:
    """
    InfoNCE / NT-Xent simplificada (SimCLR, CLIP).

    Se normalizan ``z1`` y ``z2`` (L2), se calculan las similitudes coseno
    ``S = z1 z2ᵀ / τ`` y se aplica cross-entropy con la diagonal como clase
    correcta: la pareja ``(z1[i], z2[i])`` es positiva y el resto del lote son
    negativos. Si ``symmetric`` es ``True`` se promedia con la dirección
    ``z2 → z1`` (como en CLIP).

    Args:
        z1: embeddings de la vista 1. Dimensions: [batch, dim].
        z2: embeddings de la vista 2. Dimensions: [batch, dim].
        temperature: temperatura ``τ > 0``.
        symmetric: si se promedian ambas direcciones.

    Returns:
        Pérdida escalar. Dimensions: [].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class DistillationLoss(nn.Module):
    """
    Pérdida de knowledge distillation estilo examen.

    ``L = α · CE(z_s, y) + (1 - α) · T² · KL(softmax(z_t / T) ‖ softmax(z_s / T))``.
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 4.0) -> None:
        """
        Constructor.

        Args:
            alpha: peso ``α ∈ [0, 1]`` de la parte dura (etiquetas reales).
            temperature: temperatura ``T > 0`` de la parte suave.

        Raises:
            ValueError: si ``alpha`` no está en ``[0, 1]`` o ``temperature ≤ 0``.
        """

        super().__init__()

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def _hard_loss(self, student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy con las etiquetas reales.

        Args:
            student_logits: logits del estudiante. Dimensions: [batch, classes].
            labels: clase real ``int64``. Dimensions: [batch].

        Returns:
            Pérdida escalar (media). Dimensions: [].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def _soft_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        ``T² · KL`` con reducción ``batchmean``; el profesor no recibe gradiente.

        Args:
            student_logits: logits del estudiante. Dimensions: [batch, classes].
            teacher_logits: logits del profesor. Dimensions: [batch, classes].

        Returns:
            Pérdida escalar. Dimensions: [].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combina la parte dura y la suave.

        Args:
            student_logits: logits del estudiante. Dimensions: [batch, classes].
            teacher_logits: logits del profesor. Dimensions: [batch, classes].
            labels: clase real ``int64``. Dimensions: [batch].

        Returns:
            Pérdida escalar. Dimensions: [].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class CombinedSegmentationLoss(nn.Module):
    """
    ``L = λ_ce · CE + λ_dice · Dice`` para segmentación semántica.

    La CE usa ``ignore_index`` (p.ej. 255 en los bordes de Pascal VOC) y pesos de
    clase opcionales; el Dice ignora los mismos píxeles.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = 255,
        class_weights: torch.Tensor | None = None,
        smooth: float = 1.0,
    ) -> None:
        """
        Constructor.

        Args:
            ce_weight: peso ``λ_ce`` de la cross-entropy.
            dice_weight: peso ``λ_dice`` del soft Dice.
            ignore_index: etiqueta de píxeles a ignorar.
            class_weights: pesos por clase para la CE o ``None``.
                Dimensions: [classes].
            smooth: suavizado del Dice.
        """

        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.smooth = smooth
        # TODO: registra los pesos como buffer (se mueven con .to(device))
        raise NotImplementedError("TODO")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida combinada.

        Args:
            logits: logits por píxel. Dimensions: [batch, classes, height, width].
            target: clase por píxel ``int64``. Dimensions: [batch, height, width].

        Returns:
            Pérdida escalar. Dimensions: [].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")
