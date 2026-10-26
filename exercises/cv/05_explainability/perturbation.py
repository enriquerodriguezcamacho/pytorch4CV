"""
CV05 · Explicabilidad: métodos de perturbación (Occlusion, Shapley, RISE) — ejercicios.
"""

import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .utils import normalize_map, select_target, target_scores

# --------------------------------------------------------------------------- #
# Occlusion                                                                   #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def occlusion_map(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int | None = None,
    patch: int = 8,
    stride: int = 4,
    baseline: float = 0.0,
) -> torch.Tensor:
    """
    Occlusion (Zeiler & Fergus): desliza un parche cuadrado ``patch × patch``
    (todos los canales) con paso ``stride``, rellenándolo con ``baseline``.

    Para cada posición se acumula $|f_c(x) - f_c(x_{occ})|$ (logits) en los
    píxeles tapados y se cuenta cuántas veces se ha tapado cada píxel
    (*seen*). El resultado es ``values / seen``. Las posiciones de la esquina
    superior izquierda son ``range(0, H, stride)`` × ``range(0, W, stride)`` y el
    parche se recorta en los bordes, así que todos los píxeles se tapan al menos
    una vez.

    Args:
        model: modelo a explicar.
        x: imágenes de entrada. Dimensions: [batch, channels, height, width].
        target_class: clase a explicar; ``None`` usa la clase predicha.
        patch: lado del parche.
        stride: paso entre parches.
        baseline: valor con el que se rellena el parche.

    Returns:
        Mapa de importancia (sin normalizar). Dimensions: [batch, height, width].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# Valores de Shapley                                                          #
# --------------------------------------------------------------------------- #


def exact_shapley_values(
    value_fn: Callable[[frozenset[int]], float], n_players: int
) -> torch.Tensor:
    """
    Valores de Shapley exactos enumerando todas las coaliciones:

    $\\phi_i = \\sum_{S \\subseteq N\\setminus\\{i\\}}
    \\frac{|S|!\\,(M-|S|-1)!}{M!}\\,[v(S\\cup\\{i\\}) - v(S)]$.

    En explicabilidad cada "jugador" es una región (superpíxel) y $v(S)$ es la
    salida del modelo cuando solo se mantienen visibles las regiones de $S$.

    Args:
        value_fn: función de valor $v$ que recibe la coalición (``frozenset``).
        n_players: número de jugadores $M$.

    Returns:
        Valores de Shapley ``float32``. Dimensions: [n_players].
    """

    # TODO: usa itertools.combinations y math.factorial
    raise NotImplementedError("TODO")


def kernel_shap_weight(n_players: int, coalition_size: int) -> float:
    """
    Peso del kernel de Kernel SHAP:
    $\\pi(S) = \\frac{M-1}{\\binom{M}{|S|}\\,|S|\\,(M-|S|)}$.

    Args:
        n_players: número de jugadores $M$.
        coalition_size: tamaño $|S|$ de la coalición.

    Returns:
        Peso de la coalición; ``math.inf`` si $|S| = 0$ o $|S| = M$ (en la
        práctica esas coaliciones se fuerzan como restricciones).

    Raises:
        ValueError: si ``coalition_size`` no está en ``[0, n_players]``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# RISE                                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class Percentage:
    """Probabilidad en [0, 1]. Ya dada."""

    value: float

    def __post_init__(self) -> None:
        """
        Raises:
            ValueError: si el valor no está en [0, 1].
        """

        if not 0.0 <= self.value <= 1.0:
            raise ValueError("Percentage debe estar entre 0 y 1.")


class BaseRISE(ABC):
    """Clase base de RISE (Petsiuk et al., 2018)."""

    def __init__(
        self, model: nn.Module, mask_method: Literal["naive", "normal"] = "normal"
    ) -> None:
        """
        Constructor de la clase.

        Args:
            model: modelo a explicar.
            mask_method: ``"naive"`` (Bernoulli por píxel) o ``"normal"`` (método
                del paper).

        Raises:
            ValueError: si ``mask_method`` no es válido.
        """

        self.model = model
        self.mask_method = mask_method
        self.mask_function = self._select_mask_function()

    def _select_mask_function(self) -> Callable[..., torch.Tensor]:
        """
        Devuelve la función generadora de máscaras según ``self.mask_method``.

        Returns:
            ``self.create_mask_naive`` o ``self.create_mask``.

        Raises:
            ValueError: si ``mask_method`` no es ``"naive"`` ni ``"normal"``.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    @staticmethod
    def create_mask_naive(p: Percentage, height: int, width: int, **_) -> torch.Tensor:
        """
        Máscara binaria con una Bernoulli(p) independiente por píxel.

        Args:
            p: probabilidad de que un píxel se **mantenga** (valor 1).
            height: alto de la máscara.
            width: ancho de la máscara.

        Returns:
            Máscara de ceros y unos. Dimensions: [1, height, width].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    @staticmethod
    def create_mask(p: Percentage, height: int, width: int, s: int) -> torch.Tensor:
        """
        Máscara del paper de RISE:

        1. Rejilla binaria pequeña ``s × s`` con Bernoulli(p).
        2. Tamaño de celda $C_H = \\lceil H/s \\rceil$, $C_W = \\lceil W/s \\rceil$;
           *upsampling* **bilineal** a $((s+1) C_H, (s+1) C_W)$.
        3. Recorte aleatorio ``height × width`` con desplazamiento
           ``dy ∈ [0, C_H)``, ``dx ∈ [0, C_W)`` (en este orden: ``dy`` y luego
           ``dx`` con ``torch.randint``).

        El resultado es suave (valores intermedios entre 0 y 1).

        Args:
            p: probabilidad de que una celda se mantenga.
            height: alto de la máscara final.
            width: ancho de la máscara final.
            s: lado de la rejilla pequeña.

        Returns:
            Máscara con valores en [0, 1]. Dimensions: [1, height, width].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    @torch.no_grad()
    def _target_class(self, x: torch.Tensor, target_class: int | None) -> int:
        """
        Clase a explicar: ``target_class`` o la predicha sobre la imagen limpia.
        Ya dado.

        Args:
            x: imagen. Dimensions: [channels, height, width].
            target_class: clase pedida o ``None``.

        Returns:
            Índice de la clase.
        """

        self.model.eval()
        return int(select_target(self.model(x.unsqueeze(0)), target_class).item())

    @abstractmethod
    def explain(
        self,
        x: torch.Tensor,
        *,
        n_steps: int = 150,
        s: int = 7,
        p: Percentage = Percentage(0.5),
        target_class: int | None = None,
    ) -> torch.Tensor:
        """
        Calcula la explicación RISE:

        $S(\\lambda) = \\frac{\\sum_i f_c(x \\odot M_i)\\, M_i(\\lambda)}
        {\\sum_i M_i(\\lambda)}$, con $f_c$ la **probabilidad** (softmax) de la
        clase, y finalmente se normaliza min-max.

        Args:
            x: imagen. Dimensions: [channels, height, width].
            n_steps: número de máscaras $N$ (= número de *forwards*).
            s: lado de la rejilla pequeña de las máscaras.
            p: probabilidad de mantener una celda.
            target_class: clase a explicar; ``None`` usa la clase predicha.

        Returns:
            Explicación normalizada en [0, 1]. Dimensions: [height, width].
        """


class RISE(BaseRISE):
    """RISE con un bucle de ``n_steps`` *forwards* de una sola imagen."""

    @torch.no_grad()
    def explain(
        self,
        x: torch.Tensor,
        *,
        n_steps: int = 150,
        s: int = 7,
        p: Percentage = Percentage(0.5),
        target_class: int | None = None,
    ) -> torch.Tensor:
        """
        Calcula la explicación RISE con un bucle (una máscara por iteración).

        Args:
            x: imagen. Dimensions: [channels, height, width].
            n_steps: número de máscaras.
            s: lado de la rejilla pequeña de las máscaras.
            p: probabilidad de mantener una celda.
            target_class: clase a explicar; ``None`` usa la clase predicha.

        Returns:
            Explicación normalizada en [0, 1]. Dimensions: [height, width].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class VectorizedRISE(BaseRISE):
    """RISE vectorizado: procesa las máscaras en lotes de ``batch_size``."""

    @torch.no_grad()
    def explain(
        self,
        x: torch.Tensor,
        *,
        batch_size: int = 16,
        n_steps: int = 150,
        s: int = 7,
        p: Percentage = Percentage(0.5),
        target_class: int | None = None,
    ) -> torch.Tensor:
        """
        Calcula la explicación RISE por lotes. Con la misma semilla debe dar el
        **mismo** resultado que :class:`RISE` (genera las máscaras en el mismo
        orden) para cualquier ``batch_size``, aunque ``n_steps`` no sea múltiplo
        de ``batch_size``.

        Args:
            x: imagen. Dimensions: [channels, height, width].
            batch_size: número de máscaras por *forward*.
            n_steps: número total de máscaras.
            s: lado de la rejilla pequeña de las máscaras.
            p: probabilidad de mantener una celda.
            target_class: clase a explicar; ``None`` usa la clase predicha.

        Returns:
            Explicación normalizada en [0, 1]. Dimensions: [height, width].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")
