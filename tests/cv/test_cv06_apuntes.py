"""
Tests de los ejercicios del tema 6 de los apuntes.
"""

import torch
from helpers import DummyCNN, assert_close, load
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

a = load("cv/06_adversarial_attacks/apuntes.py")


class MeanThreshold(nn.Module):
    """Clase 1 si la media de la imagen supera 0.5; cuenta las consultas."""

    def __init__(self) -> None:
        super().__init__()
        self.queries = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.queries += 1
        score = x.flatten(1).mean(dim=1) - 0.5
        return torch.stack([-score, score], dim=1)


def test_pgd_stays_in_ball_and_range() -> None:
    """La salida respeta la bola ε y [0, 1], y no tiene grafo."""

    model = DummyCNN().eval()
    x, y = torch.rand(2, 3, 8, 8), torch.tensor([0, 1])
    x_adv = a.PGDAttack(model, 0.03, 0.01, 5).perturb(x, y)
    assert (x_adv - x).abs().max() <= 0.03 + 1e-6
    assert x_adv.min() >= 0 and x_adv.max() <= 1
    assert not x_adv.requires_grad
    assert all(param.grad is None for param in model.parameters())


def test_pgd_runs_every_iteration_and_random_start() -> None:
    """Ejecuta las T iteraciones y dos ejecuciones difieren por el inicio aleatorio."""

    model = MeanThreshold()
    x, y = torch.full((1, 3, 4, 4), 0.6), torch.tensor([1])
    attack = a.PGDAttack(model, 0.2, 0.05, 7)
    first, second = attack.perturb(x, y), attack.perturb(x, y)
    assert model.queries == 14
    assert not torch.equal(first, second)


def test_pgd_lowers_accuracy() -> None:
    """Sobre un modelo lineal, el ataque cambia la predicción."""

    model = MeanThreshold()
    x, y = torch.full((1, 3, 4, 4), 0.6), torch.tensor([1])
    x_adv = a.PGDAttack(model, 0.3, 0.1, 10).perturb(x, y)
    assert model(x_adv).argmax(dim=1).item() == 0


def test_boundary_attack_gets_closer_and_keeps_target() -> None:
    """Se acerca al original sin perder la clase ni superar el presupuesto."""

    model = MeanThreshold()
    x, x_init = torch.full((3, 8, 8), 0.2), torch.full((3, 8, 8), 0.9)
    result = a.boundary_attack(model, x, x_init, 1, 200)
    assert model.queries == 200
    assert (result - x).norm() < (x_init - x).norm()
    assert model(result.unsqueeze(0)).argmax(dim=1).item() == 1
    assert result.min() >= 0 and result.max() <= 1


def test_boundary_attack_rejects_when_target_is_lost() -> None:
    """Si ningún candidato conserva la clase objetivo, devuelve la imagen inicial."""

    class AlwaysZero(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[1.0, 0.0]]).expand(x.shape[0], 2)

    x, x_init = torch.zeros(3, 4, 4), torch.ones(3, 4, 4)
    assert_close(a.boundary_attack(AlwaysZero(), x, x_init, 1, 20), x_init)


def test_adversarial_training_epoch_updates_model() -> None:
    """Actualiza los pesos, deja el modelo en ``train`` y devuelve la pérdida media."""

    model = DummyCNN()
    before = model.fc.weight.data.clone()
    data = TensorDataset(torch.rand(4, 3, 8, 8), torch.tensor([0, 1, 2, 0]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    attack = a.PGDAttack(model, 0.03, 0.01, 2)
    loss = a.pgd_adversarial_training_epoch(
        model, DataLoader(data, batch_size=2), optimizer, attack
    )
    assert loss > 0 and model.training
    assert torch.norm(model.fc.weight.data - before) > 0


def test_robustness_curve_decreases() -> None:
    """Con ε = 0 es la *clean accuracy* y no mejora al aumentar ε."""

    x = torch.cat([torch.full((4, 3, 4, 4), 0.6), torch.full((4, 3, 4, 4), 0.4)])
    y = torch.cat([torch.ones(4), torch.zeros(4)]).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    curve = a.robustness_curve(MeanThreshold(), loader, [0.0, 0.05, 0.3])
    assert curve[0] == 1.0
    assert curve[2] == 0.0
    assert curve == sorted(curve, reverse=True)
