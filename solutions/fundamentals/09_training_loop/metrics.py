"""
F09 · Bucle de entrenamiento (métricas) — soluciones.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Proporción de aciertos de un clasificador.

    Args:
        logits: puntuaciones (antes o después de softmax). Dimensions:
            [batch, num_classes].
        targets: clase verdadera de cada muestra. Dimensions: [batch].

    Returns:
        Accuracy en [0, 1] como ``float`` de Python.
    """

    # BEGIN SOLUTION
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()
    # END SOLUTION


def confusion_matrix(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    Matriz de confusión **sin bucles**.

    ``cm[t, p]`` cuenta cuántas muestras de la clase verdadera ``t`` se predijeron
    como ``p`` (filas = verdad, columnas = predicción, como en scikit-learn).
    Pista: codifica cada pareja como ``t * num_classes + p`` y usa
    ``torch.bincount`` con ``minlength``.

    Args:
        preds: clase predicha. Dimensions: [num_samples].
        targets: clase verdadera. Dimensions: [num_samples].
        num_classes: número de clases ``K``.

    Returns:
        Matriz ``int64``. Dimensions: [num_classes, num_classes].
    """

    # BEGIN SOLUTION
    idx = targets.long() * num_classes + preds.long()
    counts = torch.bincount(idx, minlength=num_classes * num_classes)
    return counts.view(num_classes, num_classes)
    # END SOLUTION


def precision_recall_f1(
    cm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precision, recall y F1 **por clase** a partir de la matriz de confusión.

    $P_k = TP_k / (TP_k + FP_k)$, $R_k = TP_k / (TP_k + FN_k)$,
    $F1_k = 2 P_k R_k / (P_k + R_k)$. Si un denominador es 0, la métrica vale 0
    (como ``zero_division=0`` en scikit-learn).

    Args:
        cm: matriz de confusión (filas = verdad). Dimensions:
            [num_classes, num_classes].

    Returns:
        Precision, recall y F1 en ``float32``. Dimensions: [num_classes] cada uno.
    """

    # BEGIN SOLUTION
    cm = cm.float()
    tp = cm.diagonal()
    pred_pos = cm.sum(dim=0)  # TP + FP (columnas)
    real_pos = cm.sum(dim=1)  # TP + FN (filas)
    precision = torch.where(pred_pos > 0, tp / pred_pos.clamp_min(1), 0.0)
    recall = torch.where(real_pos > 0, tp / real_pos.clamp_min(1), 0.0)
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom.clamp_min(1e-12), 0.0)
    return precision, recall, f1
    # END SOLUTION


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """
    Evalúa un clasificador sobre un loader.

    Pone el modelo en ``eval()`` (Dropout/BatchNorm en modo inferencia) y no
    calcula gradientes. La pérdida es la media **por muestra** (ponderando cada
    lote por su tamaño, así el último lote incompleto no pesa de más), suponiendo
    que ``loss_fn`` devuelve la media del lote.

    Args:
        model: clasificador; recibe Dimensions: [batch, ...] y devuelve
            [batch, num_classes].
        loader: iterable de ``(inputs, targets)``.
        loss_fn: función de pérdida con ``reduction="mean"``.
        device: dispositivo de cómputo.

    Returns:
        Diccionario ``{"loss": float, "accuracy": float}``.
    """

    # BEGIN SOLUTION
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        bs = targets.shape[0]
        total_loss += loss_fn(logits, targets).item() * bs
        correct += (logits.argmax(dim=1) == targets).sum().item()
        n += bs
    return {"loss": total_loss / n, "accuracy": correct / n}
    # END SOLUTION
