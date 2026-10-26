"""
Utilidades compartidas por los tests.

``load`` importa un módulo de ``exercises/`` o ``solutions/`` según la opción
``--target`` de pytest (o la variable de entorno ``PYTORCH4CV_TARGET``). Así un
mismo test sirve para corregir tus ejercicios y para validar las soluciones.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(relpath: str) -> ModuleType:
    """
    Importa ``<target>/<relpath>``.

    Args:
        relpath: ruta relativa al fichero, p.ej. ``"cv/02_object_detection/iou.py"``.

    Returns:
        Módulo importado (los imports relativos dentro del tema funcionan).
    """

    target = os.environ.get("PYTORCH4CV_TARGET", "exercises")
    dotted = relpath.removesuffix(".py").replace("/", ".")
    return importlib.import_module(f"{target}.{dotted}")


def assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-5) -> None:
    """``torch.testing.assert_close`` con tolerancias razonables por defecto."""

    torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-4)


class DummyCNN(nn.Module):
    """
    CNN mínima para tests (igual de espíritu que la de ``tests/utils.py`` del examen).

    Tres ``Conv2d`` 3→3 con ReLU y una capa lineal a ``num_classes`` logits.
    Entrada: [batch, 3, height, width]. Salida: [batch, num_classes].
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.features(x)).flatten(1))
