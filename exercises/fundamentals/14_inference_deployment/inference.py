"""
F14 · Inferencia y despliegue: predicción, inferencia por lotes y exportación — ejercicios.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def predict(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Predicción de un clasificador lista para producción:

    * ``model.eval()`` (Dropout desactivado, BatchNorm con estadísticas guardadas);
    * ``torch.inference_mode()`` (sin grafo de autograd: menos memoria y más rápido);
    * se devuelve la clase predicha y su probabilidad (softmax).

    Args:
        model: clasificador.
        x: imágenes. Dimensions: [batch, C, H, W].

    Returns:
        Clases (``int64``) y probabilidades de esas clases. Dimensions: [batch] y [batch].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def batched_predict(model: nn.Module, loader: DataLoader) -> torch.Tensor:
    """
    Logits de todo un ``DataLoader``. Cada *batch* puede ser un tensor o una tupla
    ``(x, y)`` (entonces se usa solo ``x``). Modo ``eval`` e ``inference_mode``.

    Args:
        model: clasificador.
        loader: datos a predecir.

    Returns:
        Logits concatenados en el orden del *loader*. Dimensions: [N, num_classes].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def export_torchscript(model: nn.Module, example: torch.Tensor, path: str | Path) -> torch.jit.ScriptModule:
    """
    Exporta el modelo a TorchScript con ``torch.jit.trace`` (en modo ``eval``) y lo
    guarda en ``path``. El fichero se puede cargar con ``torch.jit.load`` **sin** el
    código de la clase del modelo (útil para desplegar en C++ o en otro servicio).

    Args:
        model: modelo a exportar.
        example: entrada de ejemplo para el *tracing*. Dimensions: [batch, ...].
        path: ruta del fichero ``.pt``.

    Returns:
        Modelo trazado.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def outputs_match(
    reference: nn.Module, candidate: nn.Module, x: torch.Tensor, atol: float = 1e-5
) -> bool:
    """
    Comprueba que dos modelos (p.ej. el original y el exportado) dan la misma salida
    en modo ``eval``. Es el test mínimo antes de desplegar un modelo exportado.

    Args:
        reference: modelo original.
        candidate: modelo exportado o convertido.
        x: entrada de prueba.
        atol: tolerancia absoluta.

    Returns:
        ``True`` si las salidas coinciden (``torch.allclose``).
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
