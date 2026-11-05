"""
CV01 · Transfer learning: estrategias (patrón *Template Method*) — ejercicios.

Igual que en la práctica 1: una clase abstracta con el bucle de entrenamiento
común y subclases que solo deciden qué parámetros se entrenan y con qué
optimizador.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig
from .models import freeze_backbone, get_final_layer, set_nested_attr


class TransferStrategy(ABC):
    """
    Clase base de las estrategias de *transfer learning*.

    Attributes:
        config: hiperparámetros de entrenamiento.
        loss_fn: función de pérdida (entropía cruzada).
        device: dispositivo de cálculo.
    """

    def __init__(self, config: TrainConfig | None = None) -> None:
        """
        Constructor de la clase (ya dado).

        Args:
            config: hiperparámetros; si es ``None`` se usa ``TrainConfig()``.
        """

        self.config = config if config is not None else TrainConfig()
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device(self.config.device)

    @abstractmethod
    def _prepare(self, model: nn.Module) -> tuple[nn.Module, torch.optim.Optimizer]:
        """
        Congela/descongela capas y crea el optimizador de la estrategia.

        Args:
            model: modelo con la cabeza ya adaptada al número de clases.

        Returns:
            Tupla ``(modelo a entrenar, optimizador)``.
        """

    def _train_one_epoch(
        self, model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Entrena una época completa.

        Args:
            model: modelo a entrenar.
            loader: lotes ``(inputs, labels)``. Dimensions: inputs
                [batch, ...], labels [batch].
            optimizer: optimizador con los parámetros entrenables.

        Returns:
            Pérdida media por lote de la época.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    @torch.no_grad()
    def _validate(self, model: nn.Module, loader: DataLoader) -> dict[str, float]:
        """
        Evalúa el modelo sin calcular gradientes.

        Args:
            model: modelo a evaluar.
            loader: lotes ``(inputs, labels)``.

        Returns:
            Diccionario con ``"loss"`` (media por lote) y ``"accuracy"`` (en [0, 1]).
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def execute(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader
    ) -> dict[str, list[float]]:
        """
        Ejecuta la estrategia completa (*Template Method*).

        Mueve el modelo al dispositivo, llama a ``_prepare`` y, durante
        ``config.epochs`` épocas, a ``_train_one_epoch`` y ``_validate``.

        Args:
            model: modelo con la cabeza adaptada al número de clases.
            train_loader: datos de entrenamiento.
            val_loader: datos de validación.

        Returns:
            Historial con las claves ``"train_loss"``, ``"val_loss"`` y
            ``"val_accuracy"`` (una entrada por época).
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class LastLayerStrategy(TransferStrategy):
    """
    Solo se entrena la última capa; el *backbone* queda congelado.
    """

    def _prepare(self, model: nn.Module) -> tuple[nn.Module, torch.optim.Optimizer]:
        """
        Congela el *backbone* y crea un Adam solo con los parámetros de la cabeza.

        Args:
            model: modelo con la cabeza ya adaptada.

        Returns:
            Tupla ``(modelo, optimizador)``.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class FullFineTuningStrategy(TransferStrategy):
    """
    Se entrena todo el modelo, con un *learning rate* menor en el *backbone*.
    """

    def _prepare(self, model: nn.Module) -> tuple[nn.Module, torch.optim.Optimizer]:
        """
        Descongela todo y crea un Adam con dos grupos de parámetros.

        La cabeza usa ``config.lr`` y el resto ``config.lr *
        config.backbone_lr_factor``.

        Args:
            model: modelo con la cabeza ya adaptada.

        Returns:
            Tupla ``(modelo, optimizador)``. ``optimizer.param_groups[0]`` es el
            *backbone* y ``optimizer.param_groups[1]`` la cabeza.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class FeatureExtractionStrategy(TransferStrategy):
    """
    El *backbone* se usa como extractor fijo: se calculan las *features* una sola
    vez (``no_grad``) y se entrena un clasificador lineal sobre ellas.

    Tras ``execute`` la cabeza entrenada queda dentro del modelo original, de modo
    que ``model(x)`` funciona de extremo a extremo.
    """

    def _prepare(self, model: nn.Module) -> tuple[nn.Module, torch.optim.Optimizer]:
        """
        Devuelve la cabeza (``nn.Linear``) del modelo y un Adam para ella.

        Args:
            model: modelo con la cabeza ya adaptada.

        Returns:
            Tupla ``(cabeza, optimizador)``: aquí "el modelo a entrenar" es solo
            el clasificador lineal.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def _build_feature_extractor(self, model: nn.Module) -> nn.Module:
        """
        Copia el modelo y sustituye su cabeza por ``nn.Identity`` (ya dado).

        Args:
            model: modelo de clasificación.

        Returns:
            Extractor de *features* en modo ``eval``. Salida: [batch, in_features].
        """

        extractor = copy.deepcopy(model)
        name, _ = get_final_layer(extractor)
        set_nested_attr(extractor, name, nn.Identity())
        return extractor.eval()

    @torch.no_grad()
    def _extract_features(
        self, extractor: nn.Module, loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pasa todo el ``loader`` por el extractor sin gradientes.

        Args:
            extractor: *backbone* sin cabeza (en modo ``eval``).
            loader: lotes ``(inputs, labels)``.

        Returns:
            Tupla ``(features, labels)``. Dimensions: [num_samples, in_features] y
            [num_samples].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def execute(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader
    ) -> dict[str, list[float]]:
        """
        Extrae *features* de train y val y entrena la cabeza sobre ellas.

        Pasos: mover el modelo al dispositivo, ``_prepare`` (cabeza + optimizador),
        construir el extractor, extraer *features* de ambos *loaders*, crear
        ``DataLoader`` de ``TensorDataset`` (``config.feature_batch_size``; el de
        train con ``shuffle=True``) y repetir ``_train_one_epoch`` / ``_validate``
        sobre la cabeza.

        Args:
            model: modelo con la cabeza adaptada al número de clases.
            train_loader: datos de entrenamiento (imágenes).
            val_loader: datos de validación (imágenes).

        Returns:
            Historial con ``"train_loss"``, ``"val_loss"`` y ``"val_accuracy"``.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")
