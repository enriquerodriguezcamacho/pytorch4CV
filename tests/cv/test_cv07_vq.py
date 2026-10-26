"""
Tests de CV07 · VQ-VAE (cuantizador vectorial).
"""

import torch
import torch.nn.functional as F
from helpers import assert_close, load

m = load("cv/07_generative_models/vq.py")

NUM_EMBEDDINGS = 3
EMBEDDING_DIM = 2
CODEBOOK = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-2.0, 0.0]])


def _quantizer(beta: float = 0.25) -> torch.nn.Module:
    vq = m.VectorQuantizer(NUM_EMBEDDINGS, EMBEDDING_DIM, beta)
    with torch.no_grad():
        vq.codebook.weight.copy_(CODEBOOK)
    return vq


def test_nearest_codebook_indices() -> None:
    """Índices del vecino más cercano a mano y contra cdist + argmin."""

    z = torch.tensor([[0.9, 1.2], [-1.5, 0.1], [0.1, -0.2]])
    idx = m.nearest_codebook_indices(z, CODEBOOK)
    assert idx.dtype == torch.long, "Los índices son int64."
    assert torch.equal(idx, torch.tensor([1, 2, 0])), f"Índices {idx.tolist()}"
    z = torch.randn(50, 4)
    cb = torch.randn(7, 4)
    ref = ((z[:, None] - cb[None]) ** 2).sum(-1).argmin(1)
    assert torch.equal(m.nearest_codebook_indices(z, cb), ref), "No coincide con la ref."


def test_vq_losses_values_and_stop_gradient() -> None:
    """Mismo valor numérico, pero cada pérdida mueve solo su parte."""

    z_e = torch.tensor([[1.0, 3.0]], requires_grad=True)
    z_q = torch.tensor([[0.0, 1.0]], requires_grad=True)
    cb_loss, commit = m.vq_losses(z_e, z_q)
    assert torch.isclose(cb_loss, torch.tensor(2.5)) and torch.isclose(commit, cb_loss), (
        "Ambas son MSE((1, 3), (0, 1)) = (1 + 4) / 2 = 2.5."
    )
    cb_loss.backward()
    assert z_e.grad is None, "codebook_loss no debe dar gradiente al encoder (sg[z_e])."
    assert z_q.grad is not None, "codebook_loss mueve el codebook."
    z_q.grad = None
    commit.backward()
    assert z_q.grad is None, "commitment_loss no debe mover el codebook (sg[z_q])."
    assert z_e.grad is not None, "commitment_loss mueve el encoder."


def test_vector_quantizer_forward() -> None:
    """Formas, índices espaciales y valores cuantizados."""

    vq = _quantizer()
    z_e = torch.tensor([[[[0.9, -1.5], [0.1, 2.0]], [[1.2, 0.1], [-0.2, 2.0]]]])
    z_q, loss, idx = vq(z_e)
    assert z_q.shape == z_e.shape, f"z_q con forma {z_q.shape}"
    assert idx.shape == (1, 2, 2), f"Índices con forma {idx.shape}"
    assert torch.equal(idx[0], torch.tensor([[1, 2], [0, 1]])), f"Índices {idx[0]}"
    expected = CODEBOOK[idx[0]].permute(2, 0, 1)[None]
    assert_close(z_q, expected)
    mse = F.mse_loss(expected, z_e)
    assert torch.isclose(loss, mse * 1.25), "loss = codebook + 0.25 * commitment."


def test_vector_quantizer_beta_scaling() -> None:
    """Solo la commitment loss escala con beta."""

    z_e = torch.randn(2, EMBEDDING_DIM, 3, 3)
    _, loss0, _ = _quantizer(0.0)(z_e)
    _, loss1, _ = _quantizer(1.0)(z_e)
    _, loss_half, _ = _quantizer(0.5)(z_e)
    assert torch.isclose(loss1, 2 * loss0), "Con beta=1: codebook + commitment = 2·MSE."
    assert torch.isclose(loss_half, 1.5 * loss0), "Con beta=0.5: 1.5·MSE."


def test_straight_through_estimator() -> None:
    """El gradiente de z_q respecto a z_e es la identidad (se copia del decoder)."""

    vq = _quantizer()
    z_e = torch.randn(2, EMBEDDING_DIM, 3, 3, requires_grad=True)
    z_q, _, _ = vq(z_e)
    upstream = torch.randn_like(z_q)
    (z_q * upstream).sum().backward()
    assert z_e.grad is not None, "Sin straight-through el gradiente no llega al encoder."
    assert_close(z_e.grad, upstream)
    assert vq.codebook.weight.grad is None or vq.codebook.weight.grad.abs().sum() == 0, (
        "La reconstrucción no debe mover el codebook (solo lo hace la pérdida VQ)."
    )


def test_codebook_learns_with_vq_loss() -> None:
    """Minimizando la pérdida VQ, los códigos usados se acercan a los z_e."""

    vq = m.VectorQuantizer(4, EMBEDDING_DIM)
    opt = torch.optim.SGD(vq.parameters(), lr=0.5)
    z_e = torch.randn(4, EMBEDDING_DIM, 2, 2)
    _, first, _ = vq(z_e)
    for _ in range(20):
        _, loss, _ = vq(z_e)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss < first, "La pérdida VQ debe bajar al actualizar el codebook."
