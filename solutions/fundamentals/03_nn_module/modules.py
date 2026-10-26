"""
F03 · nn.Module — soluciones.

Ejercicios sobre cómo se construyen, inspeccionan, inicializan, congelan,
guardan y modifican los modelos de PyTorch.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# 1. Capas y modelos propios
# ---------------------------------------------------------------------------


class MyLinear(nn.Module):
    """
    Capa lineal propia ``y = x Wᵀ + b`` con ``nn.Parameter``.

    Inicialización idéntica a ``nn.Linear`` (y en el mismo orden, para que con la
    misma semilla salgan los mismos pesos):

    * ``weight ~ kaiming_uniform_(a=√5)`` ⇔ ``U(-1/√fan_in, 1/√fan_in)``.
    * ``bias ~ U(-1/√fan_in, 1/√fan_in)``.

    Si ``bias=False`` registra ``bias`` como ``None`` con ``register_parameter``.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """
        Args:
            in_features: dimensión de entrada.
            out_features: dimensión de salida.
            bias: si se usa sesgo.
        """

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # BEGIN SOLUTION
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        # END SOLUTION
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Inicializa ``weight`` y ``bias`` como ``nn.Linear``."""

        # BEGIN SOLUTION
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [*, in_features].

        Returns:
            Salida. Dimensions: [*, out_features].
        """

        # BEGIN SOLUTION
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
        # END SOLUTION

    def extra_repr(self) -> str:
        """Texto que aparece en ``print(model)`` (ya dado)."""

        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class MLP(nn.Module):
    """
    Perceptrón multicapa configurable.

    Para cada tamaño oculto ``h`` añade a ``self.layers`` (un ``nn.ModuleList``)
    un bloque ``nn.Sequential(nn.Linear(prev, h), activation(), nn.Dropout(dropout))``.
    La última capa ``self.head = nn.Linear(prev, out_features)`` no lleva
    activación (devuelve logits).
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: list[int],
        out_features: int,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            in_features: dimensión de entrada.
            hidden_sizes: tamaños de las capas ocultas (puede ser vacía).
            out_features: dimensión de salida.
            activation: clase de la activación (se instancia una por bloque).
            dropout: probabilidad de dropout tras cada activación.
        """

        super().__init__()

        # BEGIN SOLUTION
        self.layers = nn.ModuleList()
        prev = in_features
        for h in hidden_sizes:
            self.layers.append(
                nn.Sequential(nn.Linear(prev, h), activation(), nn.Dropout(dropout))
            )
            prev = h
        self.head = nn.Linear(prev, out_features)
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, in_features].

        Returns:
            Logits. Dimensions: [batch, out_features].
        """

        # BEGIN SOLUTION
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
        # END SOLUTION


class MultiTaskHead(nn.Module):
    """
    Varias cabezas lineales sobre las mismas *features*, guardadas en ``nn.ModuleDict``.

    Ejemplo: ``MultiTaskHead(64, {"class": 10, "bbox": 4})`` → ``forward`` devuelve
    ``{"class": [B, 10], "bbox": [B, 4]}``.
    """

    def __init__(self, in_features: int, tasks: dict[str, int]) -> None:
        """
        Args:
            in_features: dimensión de las *features* compartidas.
            tasks: nombre de la tarea → dimensión de salida.
        """

        super().__init__()

        # BEGIN SOLUTION
        self.heads = nn.ModuleDict(
            {name: nn.Linear(in_features, dim) for name, dim in tasks.items()}
        )
        # END SOLUTION

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: *features* compartidas. Dimensions: [batch, in_features].

        Returns:
            Diccionario tarea → salida. Dimensions: [batch, dim_tarea].
        """

        # BEGIN SOLUTION
        return {name: head(features) for name, head in self.heads.items()}
        # END SOLUTION


class RunningMean(nn.Module):
    """
    Centra los datos con una media móvil guardada como **buffer** (como BatchNorm).

    * ``running_mean`` (buffer, ceros al inicio) y ``num_batches_tracked``
      (buffer ``long``, 0 al inicio) se guardan en el ``state_dict`` y se mueven
      con ``.to(device)``, pero **no** son parámetros (no los ve el optimizador).
    * En ``train``: actualiza ``running_mean ← (1 - momentum)·running_mean +
      momentum·mean_batch`` sin grafo, incrementa ``num_batches_tracked`` y
      devuelve ``x - mean_batch``.
    * En ``eval``: devuelve ``x - running_mean`` sin actualizar nada.
    """

    def __init__(self, num_features: int, momentum: float = 0.1) -> None:
        """
        Args:
            num_features: número de características.
            momentum: peso de la media del lote en la actualización.
        """

        super().__init__()
        self.momentum = momentum

        # BEGIN SOLUTION
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, num_features].

        Returns:
            Entrada centrada. Dimensions: [batch, num_features].
        """

        # BEGIN SOLUTION
        if self.training:
            batch_mean = x.mean(dim=0)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(
                    self.momentum * batch_mean
                )
                self.num_batches_tracked += 1
            return x - batch_mean
        return x - self.running_mean
        # END SOLUTION


# ---------------------------------------------------------------------------
# 2. Inspeccionar, congelar e inicializar
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Cuenta los parámetros (escalares) de un modelo.

    Args:
        model: modelo.
        trainable_only: si ``True``, solo los que tienen ``requires_grad``.

    Returns:
        Número total de escalares.
    """

    # BEGIN SOLUTION
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )
    # END SOLUTION


def freeze_all_but_head(model: nn.Module, head_name: str) -> nn.Module:
    """
    Congela todo el modelo salvo el submódulo ``head_name`` (estrategia *LastLayer*).

    Args:
        model: modelo.
        head_name: nombre (con puntos si está anidado, p.ej. ``"classifier.6"``)
            del submódulo que se entrena.

    Returns:
        El mismo modelo.

    Raises:
        AttributeError: si no existe el submódulo (lo lanza ``get_submodule``).
    """

    # BEGIN SOLUTION
    head = model.get_submodule(head_name)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in head.parameters():
        p.requires_grad_(True)
    return model
    # END SOLUTION


def init_weights(module: nn.Module) -> None:
    """
    Inicialización para usar con ``model.apply(init_weights)``.

    * ``nn.Conv2d``: ``kaiming_normal_(mode="fan_out", nonlinearity="relu")``
      (como torchvision en ResNet) y sesgo a cero.
    * ``nn.Linear``: ``xavier_uniform_`` y sesgo a cero.
    * ``nn.BatchNorm2d`` / ``nn.GroupNorm`` / ``nn.LayerNorm``: peso 1 y sesgo 0.
    * Otros módulos: no se tocan.

    Args:
        module: un módulo cualquiera (``apply`` lo llama con cada submódulo).
    """

    # BEGIN SOLUTION
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    # END SOLUTION


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """
    Pone el modelo en ``eval`` pero deja los ``nn.Dropout`` en ``train`` (MC Dropout).

    Así BatchNorm usa sus estadísticas guardadas, mientras que cada forward
    muestrea una máscara de dropout distinta (estimación de incertidumbre).

    Args:
        model: modelo.

    Returns:
        El mismo modelo.
    """

    # BEGIN SOLUTION
    model.eval()
    for mod in model.modules():
        if isinstance(mod, (nn.Dropout, nn.Dropout2d, nn.Dropout1d, nn.Dropout3d)):
            mod.train()
    return model
    # END SOLUTION


# ---------------------------------------------------------------------------
# 3. Guardar, cargar y modificar
# ---------------------------------------------------------------------------


def save_weights(model: nn.Module, path: str | Path) -> None:
    """
    Guarda **solo** el ``state_dict`` del modelo.

    Args:
        model: modelo.
        path: ruta del fichero ``.pt``.
    """

    # BEGIN SOLUTION
    torch.save(model.state_dict(), path)
    # END SOLUTION


def load_weights(model: nn.Module, path: str | Path, strict: bool = True) -> nn.Module:
    """
    Carga un ``state_dict`` en CPU de forma segura (``weights_only=True``).

    Args:
        model: modelo con la misma arquitectura.
        path: ruta del fichero.
        strict: si ``True``, las claves deben coincidir exactamente.

    Returns:
        El mismo modelo con los pesos cargados.
    """

    # BEGIN SOLUTION
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=strict)
    return model
    # END SOLUTION


def replace_last_layer(model: nn.Module, num_classes: int) -> nn.Module:
    """
    Sustituye la **última** ``nn.Linear`` del modelo por una nueva con ``num_classes`` salidas.

    Funciona con ``resnet`` (``fc``), ``vgg`` (``classifier.6``),
    ``mobilenet_v3`` (``classifier.3``)... Busca la última ``nn.Linear`` en
    ``named_modules()`` y la reemplaza en su módulo padre (``setattr`` sirve
    también para índices de ``nn.Sequential``). La capa nueva conserva
    ``in_features`` y el uso de sesgo, y se crea en el mismo dispositivo.

    Args:
        model: modelo.
        num_classes: número de clases del nuevo problema.

    Returns:
        El mismo modelo modificado.

    Raises:
        ValueError: si el modelo no tiene ninguna ``nn.Linear``.
    """

    # BEGIN SOLUTION
    last_name = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last_name = name
    if last_name is None:
        raise ValueError("El modelo no contiene ninguna nn.Linear")
    old = model.get_submodule(last_name)
    new = nn.Linear(
        old.in_features,
        num_classes,
        bias=old.bias is not None,
        device=old.weight.device,
    )
    parent_name, _, child_name = last_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, new)
    return model
    # END SOLUTION


def functional_mlp(x: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    MLP de dos capas escrito con ``torch.nn.functional`` (sin módulos).

    ``out = linear(relu(linear(x, w1, b1)), w2, b2)``.

    Args:
        x: entrada. Dimensions: [batch, in_features].
        params: ``{"w1": [hidden, in], "b1": [hidden], "w2": [out, hidden], "b2": [out]}``.

    Returns:
        Salida. Dimensions: [batch, out].
    """

    # BEGIN SOLUTION
    h = F.relu(F.linear(x, params["w1"], params["b1"]))
    return F.linear(h, params["w2"], params["b2"])
    # END SOLUTION
