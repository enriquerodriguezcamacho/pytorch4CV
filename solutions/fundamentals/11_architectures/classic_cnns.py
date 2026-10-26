"""
F11 · Arquitecturas: redes completas — soluciones.

MLP, LeNet-5, una ResNet pequeña y un constructor de CNN a partir de una lista de
configuración (como la ``architecture_config`` de YOLOv1 de la práctica 2).
"""

import torch
from torch import nn

from .blocks import ResidualBlock

# Ejemplo de configuración al estilo YOLOv1 (backbone "Darknet" reducido):
# tupla = (kernel_size, out_channels, stride, padding); "M" = MaxPool 2×2 / 2;
# lista = [tupla_1, tupla_2, ..., num_repeats].
TINY_CONFIG: list = [
    (7, 16, 2, 3),
    "M",
    (3, 32, 1, 1),
    "M",
    (1, 16, 1, 0),
    (3, 32, 1, 1),
    "M",
    [(1, 16, 1, 0), (3, 32, 1, 1), 2],
    (3, 64, 2, 1),
]


class MLP(nn.Module):
    """
    Perceptrón multicapa: aplana la entrada y aplica ``Linear → ReLU → [Dropout]``
    por cada capa oculta y una ``Linear`` final (logits, sin activación).

    Atributo que hay que crear: ``self.net`` (``nn.Sequential`` que empieza por
    ``nn.Flatten()``).
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: list[int],
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        """
        Constructor.

        Args:
            in_features: número de características de la entrada aplanada.
            hidden_sizes: tamaños de las capas ocultas (puede estar vacía → regresión
                logística multiclase).
            num_classes: número de salidas.
            dropout: probabilidad de dropout tras cada ReLU (0 → sin dropout).
        """

        super().__init__()

        # BEGIN SOLUTION
        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_features
        for size in hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = size
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, *] con ``in_features`` elementos por muestra.

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # BEGIN SOLUTION
        return self.net(x)
        # END SOLUTION


class LeNet5(nn.Module):
    """
    LeNet-5 (LeCun et al., 1998) para imágenes de 32×32.

    ``self.features``: Conv 5×5 → 6, Tanh, AvgPool 2; Conv 5×5 → 16, Tanh, AvgPool 2.
    ``self.classifier``: Flatten, Linear 400 → 120, Tanh, Linear 120 → 84, Tanh,
    Linear 84 → ``num_classes``.

    Con 1 canal y 10 clases tiene 61 706 parámetros.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        """
        Constructor.

        Args:
            in_channels: canales de la imagen.
            num_classes: número de clases.
        """

        super().__init__()

        # BEGIN SOLUTION
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: imágenes. Dimensions: [batch, in_channels, 32, 32].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # BEGIN SOLUTION
        return self.classifier(self.features(x))
        # END SOLUTION


class CNNBlock(nn.Module):
    """
    Bloque Conv (sin sesgo) + BatchNorm + LeakyReLU(0.1), como en YOLOv1 (ya dado).
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


def build_cnn_from_config(config: list, in_channels: int = 3) -> nn.Sequential:
    """
    Construye una CNN a partir de una lista de configuración (estilo YOLOv1).

    Elementos admitidos:
        * ``tuple`` ``(kernel_size, out_channels, stride, padding)`` → ``CNNBlock``.
        * ``"M"`` → ``nn.MaxPool2d(kernel_size=2, stride=2)``.
        * ``list`` ``[tupla_1, ..., tupla_n, num_repeats]`` → repite ``num_repeats``
          veces la secuencia de ``CNNBlock``.

    Args:
        config: lista de configuración.
        in_channels: canales de la imagen de entrada.

    Returns:
        ``nn.Sequential`` con los bloques en orden.

    Raises:
        ValueError: si algún elemento no es de los tipos admitidos.
    """

    # BEGIN SOLUTION
    layers: list[nn.Module] = []
    channels = in_channels
    for item in config:
        if isinstance(item, tuple):
            k, out_ch, s, p = item
            layers.append(CNNBlock(channels, out_ch, k, s, p))
            channels = out_ch
        elif item == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif isinstance(item, list):
            *convs, repeats = item
            for _ in range(repeats):
                for k, out_ch, s, p in convs:
                    layers.append(CNNBlock(channels, out_ch, k, s, p))
                    channels = out_ch
        else:
            raise ValueError(f"Elemento de configuración no válido: {item!r}")
    return nn.Sequential(*layers)
    # END SOLUTION


class SmallResNet(nn.Module):
    """
    ResNet pequeña para imágenes tipo CIFAR.

    * ``self.stem``: Conv 3×3 → ``widths[0]`` (sin sesgo), BN, ReLU.
    * ``self.stages``: ``nn.Sequential`` con, para cada ancho ``w_i``, dos
      ``ResidualBlock``; el primero de cada etapa (salvo la primera) usa ``stride=2``.
    * ``self.pool``: ``nn.AdaptiveAvgPool2d(1)`` (global average pooling).
    * ``self.fc``: ``nn.Linear(widths[-1], num_classes)``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        widths: tuple[int, ...] = (16, 32, 64),
    ) -> None:
        """
        Constructor.

        Args:
            in_channels: canales de la imagen.
            num_classes: número de clases.
            widths: canales de cada etapa.
        """

        super().__init__()

        # BEGIN SOLUTION
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        prev = widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            blocks += [ResidualBlock(prev, w, stride), ResidualBlock(w, w)]
            prev = w
        self.stages = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1], num_classes)
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: imágenes. Dimensions: [batch, in_channels, height, width].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # BEGIN SOLUTION
        x = self.stages(self.stem(x))
        return self.fc(self.pool(x).flatten(1))
        # END SOLUTION
