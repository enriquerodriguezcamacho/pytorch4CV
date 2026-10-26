"""Configuración común de pytest.

    pytest                      # corrige tus ejercicios (exercises/)
    pytest --target solutions   # comprueba las soluciones de referencia
"""

from __future__ import annotations

import os

import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--target",
        action="store",
        default=os.environ.get("PYTORCH4CV_TARGET", "exercises"),
        choices=["exercises", "solutions"],
        help="Carpeta contra la que se ejecutan los tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    os.environ["PYTORCH4CV_TARGET"] = config.getoption("--target")


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Semilla fija en todos los tests (reproducibilidad, como en el examen)."""
    torch.manual_seed(42)
