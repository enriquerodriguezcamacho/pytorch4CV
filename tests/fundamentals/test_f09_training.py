"""
Tests de F09 · Bucle de entrenamiento.
"""

import copy
from pathlib import Path
from unittest import mock

import pytest
import torch
from helpers import assert_close, load
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

metrics = load("fundamentals/09_training_loop/metrics.py")
tr = load("fundamentals/09_training_loop/trainer.py")

IN_FEATURES = 2
NUM_CLASSES = 2
NUM_SAMPLES = 64
BATCH_SIZE = 16
LR = 0.1


# ---------------------------------------------------------------------------
# Utilidades de los tests
# ---------------------------------------------------------------------------
class RecordingMLP(nn.Module):
    """MLP diminuto que apunta ``self.training`` y si hay grad en cada forward."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_FEATURES, 16), nn.ReLU(), nn.Linear(16, NUM_CLASSES)
        )
        self.training_flags: list[bool] = []
        self.grad_flags: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.training_flags.append(self.training)
        self.grad_flags.append(torch.is_grad_enabled())
        return self.net(x)


def make_blobs(n: int = NUM_SAMPLES) -> TensorDataset:
    """Dos nubes gaussianas separables en 2D."""

    y = torch.arange(n) % 2
    centers = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
    x = centers[y] + 0.5 * torch.randn(n, IN_FEATURES)
    return TensorDataset(x, y)


def make_trainer(accumulation_steps: int = 1, shuffle: bool = True) -> "tr.Trainer":
    """Trainer con MLP, SGD y loaders sintéticos."""

    model = RecordingMLP()
    train_ds, val_ds = make_blobs(), make_blobs(32)
    return tr.Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=LR),
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=shuffle),
        val_loader=DataLoader(val_ds, batch_size=BATCH_SIZE),
        accumulation_steps=accumulation_steps,
    )


def snapshot(model: nn.Module) -> list[torch.Tensor]:
    """Copia de los parámetros."""

    return [p.detach().clone() for p in model.parameters()]


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def test_accuracy_by_hand() -> None:
    """2 de 4 aciertos → 0.5."""

    logits = torch.tensor([[2.0, 1.0], [0.0, 3.0], [1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1, 1, 0])
    acc = metrics.accuracy(logits, targets)
    assert isinstance(acc, float), "accuracy debe devolver un float."
    assert acc == pytest.approx(0.5), "Accuracy incorrecta."


def test_confusion_matrix_by_hand() -> None:
    """Filas = verdad, columnas = predicción."""

    preds = torch.tensor([0, 1, 1, 2, 2, 2])
    targets = torch.tensor([0, 0, 1, 2, 1, 2])
    cm = metrics.confusion_matrix(preds, targets, 3)
    expected = torch.tensor([[1, 1, 0], [0, 1, 1], [0, 0, 2]])
    assert cm.shape == (3, 3), "Forma incorrecta."
    assert torch.equal(cm.long(), expected), "Matriz de confusión incorrecta."
    assert cm.sum() == 6, "La suma debe ser el número de muestras."


def test_confusion_matrix_missing_classes() -> None:
    """Clases que no aparecen dan filas/columnas de ceros."""

    cm = metrics.confusion_matrix(torch.tensor([0, 0]), torch.tensor([0, 0]), 4)
    assert cm.shape == (4, 4), "Debe usar minlength."
    assert cm[0, 0] == 2 and cm.sum() == 2, "Conteos incorrectos."


def test_precision_recall_f1_by_hand() -> None:
    """Valores a mano con la matriz del test anterior."""

    cm = torch.tensor([[1, 1, 0], [0, 1, 1], [0, 0, 2]])
    p, r, f1 = metrics.precision_recall_f1(cm)
    assert_close(p, torch.tensor([1.0, 0.5, 2 / 3]))
    assert_close(r, torch.tensor([0.5, 0.5, 1.0]))
    assert_close(f1, torch.tensor([2 / 3, 0.5, 0.8]))


def test_precision_recall_f1_zero_division() -> None:
    """Clase nunca predicha ni presente → 0, sin NaN."""

    cm = torch.tensor([[3, 0], [0, 0]])
    p, r, f1 = metrics.precision_recall_f1(cm)
    assert not torch.isnan(torch.cat([p, r, f1])).any(), "No debe haber NaN."
    assert_close(p, torch.tensor([1.0, 0.0]))
    assert_close(f1, torch.tensor([1.0, 0.0]))


def test_evaluate() -> None:
    """Pérdida media por muestra, eval() y sin gradientes."""

    model = RecordingMLP()
    ds = make_blobs(10)
    loss_fn = nn.CrossEntropyLoss()
    out = metrics.evaluate(model, DataLoader(ds, batch_size=4), loss_fn)
    x, y = ds.tensors
    with torch.no_grad():
        expected = loss_fn(model.net(x), y).item()
        expected_acc = (model.net(x).argmax(1) == y).float().mean().item()
    assert out["loss"] == pytest.approx(expected, abs=1e-5), "Ponderar por tamaño."
    assert out["accuracy"] == pytest.approx(expected_acc), "Accuracy incorrecta."
    assert not any(model.training_flags), "evaluate debe usar model.eval()."
    assert not any(model.grad_flags), "evaluate debe usar torch.no_grad()."


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def test_make_epoch_train_updates_parameters() -> None:
    """Los parámetros cambian tras una época y el modo es ``train``."""

    trainer = make_trainer()
    trainer.model.eval()
    old = snapshot(trainer.model)
    loss = trainer._make_epoch_train()
    assert isinstance(loss, float) and loss > 0, "Debe devolver la pérdida media."
    for o, p in zip(old, trainer.model.parameters(), strict=True):
        assert torch.norm(p.detach() - o) > 0, "Los parámetros no se han actualizado."
    assert all(trainer.model.training_flags), "Falta model.train()."
    assert all(trainer.model.grad_flags), "En train debe haber gradientes."


def test_make_epoch_test_no_update() -> None:
    """``_make_epoch_test`` no toca pesos, usa eval() y no_grad()."""

    trainer = make_trainer()
    old = snapshot(trainer.model)
    loss = trainer._make_epoch_test()
    assert isinstance(loss, float), "Debe devolver un float."
    for o, p in zip(old, trainer.model.parameters(), strict=True):
        assert torch.equal(o, p), "La validación no debe modificar los pesos."
    assert not any(trainer.model.training_flags), "Falta model.eval()."
    assert not any(trainer.model.grad_flags), "Falta torch.no_grad()."


def test_gradient_accumulation_equivalence() -> None:
    """2 lotes de 8 acumulados == 1 lote de 16 (con SGD y pérdida media)."""

    torch.manual_seed(0)
    ds = make_blobs(16)
    model_a = RecordingMLP()
    model_b = copy.deepcopy(model_a)
    loss_fn = nn.CrossEntropyLoss()
    t_a = tr.Trainer(
        model_a,
        torch.optim.SGD(model_a.parameters(), lr=LR),
        loss_fn,
        DataLoader(ds, batch_size=8),
        DataLoader(ds, batch_size=8),
        accumulation_steps=2,
    )
    t_b = tr.Trainer(
        model_b,
        torch.optim.SGD(model_b.parameters(), lr=LR),
        loss_fn,
        DataLoader(ds, batch_size=16),
        DataLoader(ds, batch_size=16),
    )
    t_a._make_epoch_train()
    t_b._make_epoch_train()
    for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert_close(pa, pb, atol=1e-6)


@pytest.mark.parametrize("epochs", [1, 3, 5])
def test_fit_calls_epochs(epochs: int) -> None:
    """``fit`` llama ``epochs`` veces a cada método y llena el historial."""

    trainer = make_trainer()
    with (
        mock.patch.object(tr.Trainer, "_make_epoch_train", return_value=1.0) as m_tr,
        mock.patch.object(tr.Trainer, "_make_epoch_test", return_value=2.0) as m_te,
    ):
        history = trainer.fit(epochs)
    assert m_tr.call_count == epochs, "_make_epoch_train no se llamó epochs veces."
    assert m_te.call_count == epochs, "_make_epoch_test no se llamó epochs veces."
    assert history["train_loss"] == [1.0] * epochs, "Historial de train incorrecto."
    assert history["val_loss"] == [2.0] * epochs, "Historial de val incorrecto."


def test_fit_history_grows() -> None:
    """El historial crece entre llamadas sucesivas a ``fit``."""

    trainer = make_trainer()
    trainer.fit(2)
    trainer.fit(1)
    assert len(trainer.history["train_loss"]) == 3, "len(history) debe crecer."
    assert len(trainer.history["val_loss"]) == 3, "len(history) debe crecer."


def test_fit_saves_weights(tmp_path: Path) -> None:
    """Guarda ``epoch_k.pt`` cada ``save_every`` épocas y ``best.pt``."""

    trainer = make_trainer()
    folder = tmp_path / "weights"
    val_losses = iter([3.0, 1.0, 2.0, 0.5])
    with (
        mock.patch.object(tr.Trainer, "_make_epoch_train", return_value=1.0),
        mock.patch.object(
            tr.Trainer, "_make_epoch_test", side_effect=lambda: next(val_losses)
        ),
        mock.patch.object(tr.torch, "save", wraps=torch.save) as m_save,
    ):
        trainer.fit(4, save_every=2, weights_path=folder)
    files = sorted(p.name for p in folder.iterdir())
    assert files == ["best.pt", "epoch_2.pt", "epoch_4.pt"], f"Ficheros: {files}"
    # mejora en las épocas 1, 2 y 4 → 3 guardados de best + 2 periódicos
    assert m_save.call_count == 5, "best.pt solo se guarda cuando mejora."
    assert trainer.best_val_loss == 0.5, "best_val_loss incorrecto."
    state = torch.load(folder / "best.pt", weights_only=True)
    trainer.model.load_state_dict(state)


def test_fit_without_path_saves_nothing(tmp_path: Path) -> None:
    """Sin ``weights_path`` no se guarda nada."""

    trainer = make_trainer()
    with mock.patch.object(tr.torch, "save") as m_save:
        trainer.fit(2, save_every=1)
    assert m_save.call_count == 0, "No debe guardar sin weights_path."


def test_tiny_mlp_learns_separable_problem() -> None:
    """Un MLP diminuto llega a accuracy > 0.95 con datos separables."""

    trainer = make_trainer()
    history = trainer.fit(15)
    assert history["train_loss"][-1] < history["train_loss"][0], "La pérdida no baja."
    out = metrics.evaluate(trainer.model, trainer.val_loader, trainer.loss_fn)
    assert out["accuracy"] > 0.95, f"Accuracy {out['accuracy']:.2f} demasiado baja."


# ---------------------------------------------------------------------------
# Checkpoints y sanity checks
# ---------------------------------------------------------------------------
def test_save_and_load_checkpoint(tmp_path: Path) -> None:
    """Se recuperan modelo, optimizador y época."""

    model = RecordingMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x, y = make_blobs(8).tensors
    nn.CrossEntropyLoss()(model(x), y).backward()
    opt.step()
    path = tmp_path / "ckpt" / "last.pt"
    tr.save_checkpoint(path, model, opt, epoch=7, best_loss=0.3)
    assert path.exists(), "No se ha creado el fichero."
    ckpt = torch.load(path, weights_only=True)
    assert set(ckpt) >= {"model", "optimizer", "epoch", "best_loss"}, "Faltan claves."

    model2 = RecordingMLP()
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
    epoch = tr.load_checkpoint(path, model2, opt2)
    assert epoch == 7, "Época incorrecta."
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.equal(p1, p2), "Pesos no recuperados."
    assert opt2.state_dict()["state"], "Estado del optimizador no recuperado."


def test_overfit_single_batch() -> None:
    """La pérdida sobre un único lote baja casi a 0."""

    model = RecordingMLP()
    x, y = make_blobs(8).tensors
    losses = tr.overfit_single_batch(
        model, x, y, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), 0.05)
    )
    assert len(losses) == 50, "Debe devolver una pérdida por paso."
    assert losses[-1] < 0.05 < losses[0], "No consigue sobreajustar el lote."
    assert all(model.training_flags), "Debe entrenar en modo train()."


def test_set_seed_determinism() -> None:
    """``set_seed`` hace reproducible la inicialización."""

    tr.set_seed(3)
    a = nn.Linear(4, 4).weight.detach().clone()
    tr.set_seed(3)
    b = nn.Linear(4, 4).weight.detach().clone()
    assert torch.equal(a, b), "Misma semilla → mismos pesos."
