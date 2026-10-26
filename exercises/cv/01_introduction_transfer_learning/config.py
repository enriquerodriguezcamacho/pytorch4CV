"""
CV01 · Transfer learning: configuración — ejercicios.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """
    Hiperparámetros de entrenamiento de las estrategias de *transfer learning*.

    Attributes:
        epochs: número de épocas.
        lr: *learning rate* de la cabeza (y de todo el modelo en full fine-tuning).
        weight_decay: regularización L2 del optimizador.
        backbone_lr_factor: factor que multiplica a ``lr`` para el *backbone* en
            :class:`FullFineTuningStrategy` (1.0 = mismo *learning rate*).
        feature_batch_size: tamaño de lote al entrenar el clasificador lineal sobre
            las *features* extraídas.
        device: dispositivo (``"cpu"`` o ``"cuda"``).
    """

    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.0
    backbone_lr_factor: float = 0.1
    feature_batch_size: int = 32
    device: str = "cpu"


config = TrainConfig()
