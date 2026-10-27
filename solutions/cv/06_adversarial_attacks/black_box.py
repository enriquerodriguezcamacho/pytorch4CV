"""
CV06 · Ataques adversarios: ataques de caja negra (One Pixel Attack) — soluciones.

Solo se usan las **probabilidades** que devuelve el modelo (sin gradientes). Cada
individuo es un vector ``(x, y, r, g, b)`` en [0, 1]: ``x`` elige la fila
(``floor(x · H)``, recortada a ``H - 1``), ``y`` la columna y ``(r, g, b)`` el color.
Toda la aleatoriedad usa ``torch`` para que los resultados dependan de la semilla.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def apply_pixel_perturbations(img: torch.Tensor, perturbations: torch.Tensor) -> torch.Tensor:
    """
    Aplica una o varias perturbaciones de un píxel sobre una **copia** de la imagen.

    Los valores se recortan antes a [0, 1]. Se aplican en orden (si dos tocan el
    mismo píxel, gana la última).

    Args:
        img: imagen. Dimensions: [3, height, width].
        perturbations: individuos ``(x, y, r, g, b)``. Dimensions: [k, 5].

    Returns:
        Imagen perturbada. Dimensions: [3, height, width].
    """

    # BEGIN SOLUTION
    out = img.clone().detach()
    _, height, width = out.shape
    for x, y, r, g, b in perturbations.clamp(0.0, 1.0).tolist():
        row = min(int(x * height), height - 1)
        col = min(int(y * width), width - 1)
        out[:, row, col] = torch.tensor([r, g, b], dtype=out.dtype)
    return out
    # END SOLUTION


class OnePixelAttack:
    """
    One Pixel Attack (Su et al., 2019) con evolución diferencial.

    El *fitness* de un individuo es la probabilidad de la clase real tras aplicar
    su píxel: cuanto **menor**, mejor.
    """

    def __init__(
        self,
        model: nn.Module,
        img: torch.Tensor,
        label: int,
        population_size: int = 100,
        mutation_factor: float = 0.5,
    ) -> None:
        """
        Constructor de la clase (ya dado). Crea la población ``torch.rand(n, 5)``
        y la evalúa.

        Args:
            model: modelo a atacar.
            img: imagen original. Dimensions: [3, height, width].
            label: clase real.
            population_size: tamaño de la población ``n`` (``>= 4``).
            mutation_factor: factor ``F`` de la mutación.
        """

        self.model = model.eval()
        self.img = img.clone().detach()
        self.label = label
        self.n = population_size
        self.mutation_factor = mutation_factor
        self.population = torch.rand(self.n, 5)
        self.fitness = self._evaluate(self.population)
        self.history: list[float] = [self.fitness.min().item()]

    @torch.no_grad()
    def _evaluate(self, population: torch.Tensor) -> torch.Tensor:
        """
        *Fitness* de cada individuo (en un único *forward* por lotes).

        Args:
            population: individuos. Dimensions: [n, 5].

        Returns:
            Probabilidad de la clase real para cada individuo. Dimensions: [n].
        """

        # BEGIN SOLUTION
        images = torch.stack(
            [apply_pixel_perturbations(self.img, ind.unsqueeze(0)) for ind in population]
        )
        return F.softmax(self.model(images), dim=1)[:, self.label]
        # END SOLUTION

    def _mutation_indices(self) -> torch.Tensor:
        """
        Para cada individuo ``i``, tres índices **distintos entre sí y distintos de
        i** (``torch.randperm`` de los ``n - 1`` restantes). Ya dado.

        Returns:
            Índices ``(r1, r2, r3)``. Dimensions: [n, 3].
        """

        rows = []
        for i in range(self.n):
            others = torch.cat([torch.arange(i), torch.arange(i + 1, self.n)])
            rows.append(others[torch.randperm(self.n - 1)[:3]])
        return torch.stack(rows)

    def step(self) -> None:
        """
        Una generación de evolución diferencial:

        1. Mutación: ``hijo_i = x_r1 + F · (x_r2 - x_r3)``, recortado a [0, 1].
        2. Selección: el hijo sustituye al padre si su *fitness* es **estrictamente
           menor**.

        Al final añade el mejor *fitness* a ``self.history``.
        """

        # BEGIN SOLUTION
        idx = self._mutation_indices()
        p = self.population
        children = p[idx[:, 0]] + self.mutation_factor * (p[idx[:, 1]] - p[idx[:, 2]])
        children = children.clamp(0.0, 1.0)
        child_fitness = self._evaluate(children)
        better = child_fitness < self.fitness
        self.population = torch.where(better[:, None], children, self.population)
        self.fitness = torch.where(better, child_fitness, self.fitness)
        self.history.append(self.fitness.min().item())
        # END SOLUTION

    def attack(self, generations: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ejecuta ``generations`` generaciones y devuelve el mejor individuo.

        Args:
            generations: número de generaciones.

        Returns:
            Imagen perturbada con el mejor individuo y ese individuo.
            Dimensions: [3, height, width] y [5].
        """

        # BEGIN SOLUTION
        for _ in range(generations):
            self.step()
        best = self.population[self.fitness.argmin()]
        return apply_pixel_perturbations(self.img, best.unsqueeze(0)), best
        # END SOLUTION
