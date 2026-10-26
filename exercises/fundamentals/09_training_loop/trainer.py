"""
F09 · Bucle de entrenamiento (Trainer tipo examen) — ejercicios.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch
from torch import nn

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def set_seed(seed: int = 42) -> None:
    """
    Fija todas las semillas (dada).

    Args:
        seed: semilla para ``random``, NumPy y PyTorch (CPU y CUDA).
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
) -> None:
    """
    Guarda un *checkpoint* completo para poder reanudar el entrenamiento.

    Debe guardar un diccionario con las claves ``"model"`` (``state_dict`` del
    modelo), ``"optimizer"`` (``state_dict`` del optimizador), ``"epoch"`` y
    ``"best_loss"``. Crea la carpeta si no existe.

    Args:
        path: ruta del fichero ``.pt``.
        model: modelo.
        optimizer: optimizador (guarda momentos de Adam, etc.).
        epoch: última época completada.
        best_loss: mejor pérdida de validación hasta ahora.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    """
    Carga un *checkpoint* guardado con ``save_checkpoint``.

    Args:
        path: ruta del fichero ``.pt``.
        model: modelo donde cargar los pesos (se modifica *in place*).
        optimizer: optimizador donde cargar el estado (opcional).
        map_location: dispositivo al que mapear los tensores.

    Returns:
        Época guardada en el *checkpoint*.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def overfit_single_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: LossFn,
    optimizer: torch.optim.Optimizer,
    steps: int = 50,
) -> list[float]:
    """
    *Sanity check*: entrena muchas veces sobre el **mismo** lote.

    Si el modelo y el bucle están bien, la pérdida debe bajar casi a 0. Si no
    baja, hay un bug (``zero_grad`` olvidado, ``detach`` donde no toca, etiquetas
    mal...).

    Args:
        model: modelo.
        inputs: lote de entrada. Dimensions: [batch, ...].
        targets: etiquetas. Dimensions: [batch, ...].
        loss_fn: función de pérdida.
        optimizer: optimizador.
        steps: número de pasos de optimización.

    Returns:
        Lista con la pérdida (``float``) de cada paso, de longitud ``steps``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class Trainer:
    """
    Entrenador genérico al estilo del examen (``_make_epoch_train`` /
    ``_make_epoch_test`` / ``fit``).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: LossFn,
        train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        val_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        device: str | torch.device = "cpu",
        accumulation_steps: int = 1,
    ) -> None:
        """
        Constructor (dado).

        Args:
            model: modelo a entrenar.
            optimizer: optimizador sobre ``model.parameters()``.
            loss_fn: función de pérdida (``reduction="mean"``).
            train_loader: loader de entrenamiento con ``(inputs, targets)``.
            val_loader: loader de validación.
            device: dispositivo de cómputo.
            accumulation_steps: número de lotes cuyos gradientes se acumulan antes
                de cada ``optimizer.step()`` (*gradient accumulation*).
        """

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accumulation_steps = accumulation_steps
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.best_val_loss = float("inf")

    def _make_epoch_train(self) -> float:
        """
        Entrena una época completa.

        Pasos: ``model.train()``; para cada lote, mover a ``self.device``, forward,
        pérdida, ``(loss / accumulation_steps).backward()`` y, cada
        ``accumulation_steps`` lotes (y al final de la época si quedan gradientes
        pendientes), ``optimizer.step()`` + ``optimizer.zero_grad()``. Con
        ``accumulation_steps = 1`` es el bucle canónico.

        Returns:
            Pérdida media por muestra de la época (sin dividir por
            ``accumulation_steps``).
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    @torch.no_grad()
    def _make_epoch_test(self) -> float:
        """
        Evalúa una época sobre ``val_loader`` (sin gradientes, ``model.eval()``).

        Returns:
            Pérdida media por muestra.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def fit(
        self,
        epochs: int,
        save_every: int | None = None,
        weights_path: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """
        Bucle principal de entrenamiento.

        En cada época: ``_make_epoch_train``, ``_make_epoch_test`` y añadir ambas
        pérdidas a ``self.history``. Si se da ``weights_path`` (una carpeta):

        * si la pérdida de validación mejora ``self.best_val_loss``, actualizarla y
          guardar ``model.state_dict()`` en ``weights_path / "best.pt"``;
        * si además ``save_every`` no es ``None``, guardar cada ``save_every``
          épocas en ``weights_path / f"epoch_{epoch}.pt"`` (``epoch`` empieza en 1).

        Args:
            epochs: número de épocas.
            save_every: frecuencia de guardado (en épocas).
            weights_path: carpeta donde guardar los pesos (se crea si no existe).

        Returns:
            ``self.history`` con las listas ``"train_loss"`` y ``"val_loss"``.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")
