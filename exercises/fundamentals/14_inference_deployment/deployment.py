"""
F14 · Inferencia y despliegue: estrategias de despliegue del capítulo 1 — ejercicios.

Un modelo **candidato** quiere sustituir al **campeón** que está en producción:

| Estrategia | Quién responde | *Forwards* por petición | Riesgo |
|---|---|---|---|
| *Shadow* | siempre el campeón | 2 | ninguno |
| A/B testing | reparto fijo (p.ej. 50/50) | 1 | medio |
| *Canary* | poco tráfico al candidato y se va subiendo | 1 | bajo |

Cada petición es **una** imagen ``[C, H, W]``. La aleatoriedad usa un
``torch.Generator`` propio (con semilla) para que el reparto sea reproducible.
"""

from __future__ import annotations

import math

import torch
from torch import nn


@torch.inference_mode()
def _predict_one(model: nn.Module, x: torch.Tensor) -> int:
    """Clase predicha para una imagen ``[C, H, W]``. Ya dada."""

    model.eval()
    return int(model(x.unsqueeze(0)).argmax(dim=1).item())


class ShadowDeployment:
    """
    *Shadow deployment*: los dos modelos predicen, pero **solo se devuelve** la
    predicción del campeón. La del candidato se registra para compararla.

    Atributos: ``log`` (lista de pares ``(pred_campeón, pred_candidato)``).
    """

    def __init__(self, champion: nn.Module, candidate: nn.Module) -> None:
        """
        Constructor de la clase (ya dado).

        Args:
            champion: modelo en producción.
            candidate: modelo nuevo.
        """

        self.champion = champion
        self.candidate = candidate
        self.log: list[tuple[int, int]] = []

    def predict(self, x: torch.Tensor) -> int:
        """
        Atiende una petición.

        Args:
            x: imagen. Dimensions: [C, H, W].

        Returns:
            Predicción del **campeón**.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def agreement(self) -> float:
        """
        Fracción de peticiones en las que ambos modelos coinciden.

        Returns:
            Valor en [0, 1]; 0.0 si aún no hay peticiones.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class ABTest:
    """
    A/B testing: cada petición va al candidato con probabilidad ``split``
    (``torch.rand(1, generator=self.generator) < split``) y al campeón en otro caso.

    Se lleva la cuenta de peticiones y aciertos de cada modelo en ``requests`` y
    ``hits`` (diccionarios con claves ``"champion"`` y ``"candidate"``).
    """

    def __init__(
        self, champion: nn.Module, candidate: nn.Module, split: float = 0.5, seed: int = 0
    ) -> None:
        """
        Constructor de la clase (ya dado).

        Args:
            champion: modelo en producción.
            candidate: modelo nuevo.
            split: fracción de tráfico al candidato.
            seed: semilla del generador.
        """

        self.models = {"champion": champion, "candidate": candidate}
        self.split = split
        self.generator = torch.Generator().manual_seed(seed)
        self.requests = {"champion": 0, "candidate": 0}
        self.hits = {"champion": 0, "candidate": 0}

    def route(self) -> str:
        """
        Decide qué modelo atiende la siguiente petición.

        Returns:
            ``"candidate"`` o ``"champion"``.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def predict(self, x: torch.Tensor, label: int | None = None) -> tuple[str, int]:
        """
        Atiende una petición; si se conoce la etiqueta real, actualiza los aciertos.

        Args:
            x: imagen. Dimensions: [C, H, W].
            label: clase real (las etiquetas suelen llegar más tarde; ``None`` si no hay).

        Returns:
            Nombre del modelo que ha respondido y su predicción.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


def two_proportion_z_test(hits_a: int, n_a: int, hits_b: int, n_b: int) -> tuple[float, float]:
    """
    Test z de dos proporciones para decidir si la precisión de B es distinta de la
    de A (bilateral):

    ``p = (h_a + h_b) / (n_a + n_b)``, ``SE = sqrt(p (1 - p) (1/n_a + 1/n_b))``,
    ``z = (p_b - p_a) / SE`` y ``p-valor = erfc(|z| / sqrt(2))``.

    Args:
        hits_a: aciertos de A.
        n_a: peticiones de A (> 0).
        hits_b: aciertos de B.
        n_b: peticiones de B (> 0).

    Returns:
        ``(z, p_valor)``; ``(0.0, 1.0)`` si ``SE = 0``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class CanaryDeployment:
    """
    *Canary deployment*: el candidato empieza con una fracción pequeña del tráfico
    (``fractions[0]``) y se pasa a la siguiente fracción cada ``step_requests``
    peticiones, **solo si** su precisión en esa etapa es ``>= min_accuracy``. Si no
    llega, se hace *rollback*: la fracción pasa a 0 y ``rolled_back = True``.

    La ruta de cada petición se decide igual que en :class:`ABTest` con la fracción
    actual. Las peticiones de la etapa (``stage_requests``/``stage_hits``) solo
    cuentan las que atiende el candidato.
    """

    def __init__(
        self,
        champion: nn.Module,
        candidate: nn.Module,
        fractions: tuple[float, ...] = (0.05, 0.25, 0.5, 1.0),
        step_requests: int = 20,
        min_accuracy: float = 0.8,
        seed: int = 0,
    ) -> None:
        """
        Constructor de la clase (ya dado).

        Args:
            champion: modelo en producción.
            candidate: modelo nuevo.
            fractions: fracciones de tráfico de cada etapa (la última suele ser 1.0).
            step_requests: peticiones **del candidato** por etapa antes de decidir.
            min_accuracy: precisión mínima del candidato para avanzar.
            seed: semilla del generador.
        """

        self.champion = champion
        self.candidate = candidate
        self.fractions = fractions
        self.step_requests = step_requests
        self.min_accuracy = min_accuracy
        self.generator = torch.Generator().manual_seed(seed)
        self.stage = 0
        self.stage_requests = 0
        self.stage_hits = 0
        self.rolled_back = False

    @property
    def fraction(self) -> float:
        """Fracción de tráfico actual del candidato (0 tras un *rollback*)."""

        return 0.0 if self.rolled_back else self.fractions[self.stage]

    def _end_of_stage(self) -> None:
        """
        Al completar ``step_requests`` peticiones del candidato: si su precisión es
        suficiente se avanza de etapa (sin pasar de la última), si no se hace
        *rollback*. En ambos casos se reinician los contadores de la etapa.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def predict(self, x: torch.Tensor, label: int) -> tuple[str, int]:
        """
        Atiende una petición etiquetada.

        Args:
            x: imagen. Dimensions: [C, H, W].
            label: clase real.

        Returns:
            Nombre del modelo que ha respondido (``"candidate"``/``"champion"``) y
            su predicción.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")
