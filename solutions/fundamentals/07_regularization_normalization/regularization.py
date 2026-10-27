"""
F07 · Regularización y normalización — soluciones.

Dropout, BatchNorm, LayerNorm y GroupNorm desde cero (deben coincidir con los de
PyTorch), penalización L2, *label smoothing*, Mixup, CutMix (práctica 4) y
*early stopping*. Toda la aleatoriedad usa ``torch`` para depender de la semilla.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Dropout                                                                     #
# --------------------------------------------------------------------------- #


def dropout(x: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
    """
    *Inverted dropout*: en entrenamiento cada elemento se anula con probabilidad
    ``p`` y los que quedan se escalan por ``1 / (1 - p)`` (así la esperanza no
    cambia y en inferencia no hay que hacer nada). En inferencia es la identidad.

    La máscara se genera con ``torch.rand_like(x) >= p``.

    Args:
        x: entrada. Dimensions: [*].
        p: probabilidad de anular un elemento, en ``[0, 1]``.
        training: modo entrenamiento.

    Returns:
        Salida. Dimensions: [*].

    Raises:
        ValueError: si ``p`` no está en ``[0, 1]``.
    """

    # BEGIN SOLUTION
    if not 0.0 <= p <= 1.0:
        raise ValueError("p debe estar en [0, 1].")
    if not training or p == 0.0:
        return x
    if p == 1.0:
        return torch.zeros_like(x)
    mask = (torch.rand_like(x) >= p).to(x.dtype)
    return x * mask / (1.0 - p)
    # END SOLUTION


# --------------------------------------------------------------------------- #
# Normalizaciones                                                             #
# --------------------------------------------------------------------------- #


class BatchNorm2d(nn.Module):
    """
    BatchNorm 2D desde cero, equivalente a ``nn.BatchNorm2d`` (``affine=True``,
    ``track_running_stats=True``).

    * **train**: media y varianza **sesgada** por canal sobre ``(N, H, W)``;
      se normaliza con ellas y se actualizan las estadísticas
      ``running = (1 - momentum) · running + momentum · batch`` usando la
      varianza **insesgada** (``n / (n - 1)``).
    * **eval**: se normaliza con ``running_mean`` y ``running_var``.

    Salida: ``γ · (x - μ) / sqrt(σ² + eps) + β``.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        """
        Constructor de la clase.

        ``weight`` (γ, unos) y ``bias`` (β, ceros) son ``nn.Parameter``;
        ``running_mean`` (ceros) y ``running_var`` (unos) son *buffers*
        (``register_buffer``) para que se guarden en el ``state_dict`` sin ser
        parámetros entrenables.

        Args:
            num_features: número de canales ``C``.
            eps: constante de estabilidad.
            momentum: factor de actualización de las estadísticas.
        """

        super().__init__()

        self.eps = eps
        self.momentum = momentum
        # BEGIN SOLUTION
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, channels, height, width].

        Returns:
            Salida normalizada. Dimensions: igual que ``x``.
        """

        # BEGIN SOLUTION
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            n = x.numel() // x.shape[1]
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                unbiased = var * n / max(n - 1, 1)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased)
        else:
            mean, var = self.running_mean, self.running_var
        x_hat = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        return self.weight.view(1, -1, 1, 1) * x_hat + self.bias.view(1, -1, 1, 1)
        # END SOLUTION


def layer_norm(
    x: torch.Tensor,
    normalized_ndim: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    LayerNorm: normaliza cada muestra sobre sus **últimas** ``normalized_ndim``
    dimensiones (varianza sesgada), igual que ``F.layer_norm``. No depende del
    lote ni del modo train/eval (por eso se usa en Transformers).

    Args:
        x: entrada. Dimensions: [*, D_1, ..., D_k] con ``k = normalized_ndim``.
        normalized_ndim: número de dimensiones finales a normalizar.
        weight: γ opcional. Dimensions: [D_1, ..., D_k].
        bias: β opcional. Dimensions: [D_1, ..., D_k].
        eps: constante de estabilidad.

    Returns:
        Salida. Dimensions: igual que ``x``.
    """

    # BEGIN SOLUTION
    dims = tuple(range(-normalized_ndim, 0))
    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, unbiased=False, keepdim=True)
    out = (x - mean) / torch.sqrt(var + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out
    # END SOLUTION


def group_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    GroupNorm: divide los canales en ``num_groups`` grupos **consecutivos** y
    normaliza cada (muestra, grupo) sobre sus canales y posiciones. Con
    ``num_groups = 1`` es LayerNorm sobre ``(C, H, W)``; con ``num_groups = C``
    es InstanceNorm. γ y β son **por canal**.

    Args:
        x: entrada. Dimensions: [batch, channels, height, width].
        num_groups: número de grupos (divide a ``channels``).
        weight: γ opcional. Dimensions: [channels].
        bias: β opcional. Dimensions: [channels].
        eps: constante de estabilidad.

    Returns:
        Salida. Dimensions: igual que ``x``.

    Raises:
        ValueError: si ``num_groups`` no divide a ``channels``.
    """

    # BEGIN SOLUTION
    batch, channels, height, width = x.shape
    if channels % num_groups != 0:
        raise ValueError("num_groups debe dividir al número de canales.")
    grouped = x.reshape(batch, num_groups, -1)
    mean = grouped.mean(dim=2, keepdim=True)
    var = grouped.var(dim=2, unbiased=False, keepdim=True)
    out = ((grouped - mean) / torch.sqrt(var + eps)).reshape(batch, channels, height, width)
    if weight is not None:
        out = out * weight.view(1, -1, 1, 1)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out
    # END SOLUTION


# --------------------------------------------------------------------------- #
# Penalizaciones y pérdidas regularizadas                                     #
# --------------------------------------------------------------------------- #


def l2_penalty(parameters: Iterable[tuple[str, nn.Parameter]], exclude_bias: bool = True) -> torch.Tensor:
    """
    Penalización L2 ``Σ ||θ||²`` (sin el ``½``) para añadir a la pérdida
    (``loss + λ · l2_penalty(...)``). Es lo que hace ``weight_decay`` en SGD, pero
    explícito.

    Args:
        parameters: pares ``(nombre, parámetro)``, p.ej. ``model.named_parameters()``.
        exclude_bias: si se excluyen los parámetros cuyo nombre termina en
            ``"bias"`` (práctica habitual).

    Returns:
        Escalar derivable.
    """

    # BEGIN SOLUTION
    total = torch.tensor(0.0)
    for name, param in parameters:
        if exclude_bias and name.endswith("bias"):
            continue
        total = total + (param**2).sum()
    return total
    # END SOLUTION


def label_smoothing_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1
) -> torch.Tensor:
    """
    Entropía cruzada con *label smoothing* (igual que
    ``F.cross_entropy(..., label_smoothing=ε)``): el objetivo es
    ``(1 - ε) · onehot + ε / K``.

    Se calcula con ``log_softmax`` (estable) y se promedia sobre el lote.

    Args:
        logits: Dimensions: [batch, num_classes].
        targets: índices de clase. Dimensions: [batch].
        smoothing: ``ε`` en ``[0, 1]``.

    Returns:
        Escalar.
    """

    # BEGIN SOLUTION
    num_classes = logits.shape[1]
    log_probs = torch.log_softmax(logits, dim=1)
    soft = torch.full_like(log_probs, smoothing / num_classes)
    soft.scatter_add_(1, targets.view(-1, 1), torch.full_like(log_probs[:, :1], 1 - smoothing))
    return -(soft * log_probs).sum(dim=1).mean()
    # END SOLUTION


# --------------------------------------------------------------------------- #
# Mixup y CutMix                                                              #
# --------------------------------------------------------------------------- #


def sample_lambda(alpha: float) -> float:
    """
    ``λ ~ Beta(α, α)`` con ``torch.distributions`` (1.0 si ``α <= 0``). Ya dada.

    Args:
        alpha: parámetro de la Beta.

    Returns:
        ``λ`` en [0, 1].
    """

    if alpha <= 0:
        return 1.0
    beta = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(alpha))
    return float(beta.sample())


def mixup(
    x: torch.Tensor, y: torch.Tensor, lam: float, perm: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mixup (Zhang et al., 2018): ``x̃ = λ x + (1 - λ) x[perm]`` y
    ``ỹ = λ y + (1 - λ) y[perm]`` (etiquetas *one-hot* o probabilidades).

    Args:
        x: imágenes. Dimensions: [batch, channels, height, width].
        y: etiquetas *one-hot*. Dimensions: [batch, num_classes].
        lam: coeficiente ``λ`` (normalmente de :func:`sample_lambda`).
        perm: permutación del lote; si es ``None`` se usa ``torch.randperm(batch)``.

    Returns:
        ``(x̃, ỹ)``. Dimensions: iguales que ``x`` e ``y``.
    """

    # BEGIN SOLUTION
    if perm is None:
        perm = torch.randperm(x.shape[0])
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]
    # END SOLUTION


def cutmix_box(
    height: int, width: int, lam: float, cy: int, cx: int
) -> tuple[int, int, int, int]:
    """
    Caja de CutMix de área ``(1 - λ) · H · W`` centrada en ``(cy, cx)``:

    ``cut_h = round(H · sqrt(1 - λ))``, ``cut_w = round(W · sqrt(1 - λ))`` y la
    caja ``[cy - cut_h // 2, cy + cut_h // 2) × [cx - cut_w // 2, cx + cut_w // 2)``
    recortada a la imagen.

    Args:
        height: alto ``H``.
        width: ancho ``W``.
        lam: ``λ`` (proporción que se **conserva** de la primera imagen).
        cy: fila del centro.
        cx: columna del centro.

    Returns:
        ``(y1, y2, x1, x2)`` con ``y2``/``x2`` exclusivos.
    """

    # BEGIN SOLUTION
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = round(height * ratio), round(width * ratio)
    y1 = min(max(cy - cut_h // 2, 0), height)
    y2 = min(max(cy + cut_h // 2, 0), height)
    x1 = min(max(cx - cut_w // 2, 0), width)
    x2 = min(max(cx + cut_w // 2, 0), width)
    return y1, y2, x1, x2
    # END SOLUTION


def cutmix(
    images: torch.Tensor, labels: tuple[int, int], alpha: float = 1.0
) -> tuple[torch.Tensor, float]:
    """
    CutMix de una pareja de imágenes (como en la práctica 4).

    1. ``λ = sample_lambda(alpha)``.
    2. Centro uniforme: ``cy = randint(0, H)`` y luego ``cx = randint(0, W)``
       (``torch.randint``, en ese orden).
    3. Se pega en la primera imagen el recorte de la segunda dentro de
       :func:`cutmix_box`.
    4. λ se **corrige** con el área real de la caja (tras recortar a la imagen):
       ``λ' = 1 - área / (H · W)``, y la etiqueta es ``λ' · y₁ + (1 - λ') · y₂``.

    Args:
        images: pareja de imágenes. Dimensions: [2, channels, height, width].
        labels: etiquetas ``(y₁, y₂)``.
        alpha: parámetro de la Beta.

    Returns:
        Imagen mezclada y etiqueta mezclada. Dimensions: [channels, height, width].
    """

    # BEGIN SOLUTION
    _, _, height, width = images.shape
    lam = sample_lambda(alpha)
    cy = int(torch.randint(0, height, (1,)).item())
    cx = int(torch.randint(0, width, (1,)).item())
    y1, y2, x1, x2 = cutmix_box(height, width, lam, cy, cx)
    mixed = images[0].clone()
    mixed[:, y1:y2, x1:x2] = images[1, :, y1:y2, x1:x2]
    lam_adj = 1.0 - (y2 - y1) * (x2 - x1) / (height * width)
    return mixed, lam_adj * labels[0] + (1 - lam_adj) * labels[1]
    # END SOLUTION


# --------------------------------------------------------------------------- #
# Early stopping                                                              #
# --------------------------------------------------------------------------- #


class EarlyStopping:
    """
    *Early stopping* sobre una métrica que se quiere **minimizar** (p.ej. la
    pérdida de validación).

    Una época mejora si ``value < best - min_delta``; entonces se guarda
    ``best`` y se reinicia el contador. Si no, el contador sube y, cuando llega a
    ``patience``, ``should_stop`` pasa a ``True``.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        """
        Constructor de la clase (ya dado).

        Args:
            patience: épocas sin mejorar que se toleran.
            min_delta: mejora mínima para contar como mejora.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """
        Registra el valor de una época.

        Args:
            value: métrica de la época.

        Returns:
            ``True`` si esta época ha mejorado ``best``.
        """

        # BEGIN SOLUTION
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
        # END SOLUTION
