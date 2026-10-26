"""
CV03 · Style transfer: configuración del proyecto (dada) — soluciones.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class PathsConfig:
    """
    Rutas de las imágenes de entrada y de los resultados.
    """

    content_image: str = "images/content_image.png"
    style_image: str = "images/style_image.png"
    synthesized_images: str = "images/synthesized_images"


@dataclass
class ModelConfig:
    """
    Configuración del modelo y del preprocesado.
    """

    device: str = "cpu"
    size_images: int = 64
    means: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    stds: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class LossConfig:
    """
    Configuración de la pérdida. Los índices de capa se refieren al orden en el que
    se ejecutan las capas enganchadas con hooks (por defecto, las ``Conv2d``).
    """

    layers_content: list[int] = field(default_factory=lambda: [3])
    layers_style: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    style_layer_weights: dict[int, float] | None = None
    tv_squared: bool = False
    alpha: float = 0.5
    beta: float = 0.45
    gamma: float = 0.05

    def __post_init__(self) -> None:
        total = torch.tensor(self.alpha + self.beta + self.gamma)
        assert torch.isclose(total, torch.tensor(1.0)), "alpha+beta+gamma debe ser 1."


@dataclass
class TrainingConfig:
    """
    Configuración de la optimización de la imagen.
    """

    epochs: int = 50
    lr: float = 5e-2
    optimizer: type[torch.optim.Optimizer] = torch.optim.Adam
    random_synthesized_image: bool = False


@dataclass
class Config:
    """
    Configuración principal.
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


config = Config()
