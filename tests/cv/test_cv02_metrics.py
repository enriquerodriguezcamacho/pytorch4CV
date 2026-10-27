"""
Tests de CV02 · Detección de objetos: emparejamiento, curva PR, AP y mAP.
"""

import pytest
import torch
from helpers import assert_close, load

m = load("cv/02_object_detection/metrics.py")

GT = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])


def test_match_detections_tp_fp_and_duplicates() -> None:
    """Una detección buena por objeto es TP; el duplicado y la caja lejana son FP."""

    preds = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],  # TP (objeto 0)
            [0.5, 0.5, 10.0, 10.0],  # duplicado del objeto 0 → FP
            [20.0, 20.0, 30.0, 30.0],  # TP (objeto 1)
            [50.0, 50.0, 60.0, 60.0],  # no solapa → FP
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7, 0.95])
    out = m.match_detections(preds, scores, GT, 0.5)
    assert out.dtype == torch.bool, "Devuelve un tensor booleano."
    assert out.tolist() == [True, False, True, False], "Orden de entrada, no de score."


def test_match_detections_highest_score_wins() -> None:
    """El objeto se asigna a la predicción de mayor score aunque venga después."""

    preds = torch.tensor([[0.5, 0.5, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    out = m.match_detections(preds, torch.tensor([0.3, 0.9]), GT[:1], 0.5)
    assert out.tolist() == [False, True]


def test_match_detections_threshold_inclusive() -> None:
    """IoU = 0.5 exacto cuenta como TP (>=)."""

    pred = torch.tensor([[0.0, 0.0, 10.0, 5.0]])  # IoU con GT[0] = 50/100
    assert m.match_detections(pred, torch.tensor([1.0]), GT[:1], 0.5).tolist() == [True]


def test_match_detections_empty() -> None:
    assert m.match_detections(torch.zeros(0, 4), torch.zeros(0), GT).numel() == 0
    out = m.match_detections(GT, torch.tensor([0.5, 0.4]), torch.zeros(0, 4))
    assert out.tolist() == [False, False]


def test_precision_recall_curve_hand_values() -> None:
    """Ordenadas por score: TP, FP, TP con 3 objetos reales."""

    scores = torch.tensor([0.6, 0.9, 0.8])  # orden: 0.9 (TP), 0.8 (FP), 0.6 (TP)
    is_tp = torch.tensor([True, True, False])
    precision, recall = m.precision_recall_curve(scores, is_tp, num_gt=3)
    assert_close(precision, torch.tensor([1.0, 0.5, 2 / 3]))
    assert_close(recall, torch.tensor([1 / 3, 1 / 3, 2 / 3]))


def test_average_precision_all_points() -> None:
    """Envolvente monótona: p = [1, .5, 2/3] → [1, 2/3, 2/3]. AP = 1/3·1 + 1/3·2/3."""

    precision = torch.tensor([1.0, 0.5, 2 / 3])
    recall = torch.tensor([1 / 3, 1 / 3, 2 / 3])
    ap = m.average_precision(precision, recall, "all_points")
    assert isinstance(ap, float), "Devuelve un float."
    assert ap == pytest.approx(1 / 3 + 2 / 9, abs=1e-5)


def test_average_precision_11_points() -> None:
    """r ≤ 1/3 → p=1 (4 puntos: 0, .1, .2, .3); 1/3 < t ≤ 2/3 → 2/3 (3 puntos)."""

    precision = torch.tensor([1.0, 0.5, 2 / 3])
    recall = torch.tensor([1 / 3, 1 / 3, 2 / 3])
    ap = m.average_precision(precision, recall, "11_points")
    assert ap == pytest.approx((4 * 1.0 + 3 * 2 / 3) / 11, abs=1e-5)


def test_average_precision_perfect_and_invalid() -> None:
    assert m.average_precision(torch.ones(4), torch.linspace(0.25, 1, 4)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        m.average_precision(torch.ones(2), torch.ones(2), "voc")


def _perfect_case() -> tuple[list[dict], list[dict]]:
    targets = [
        {"boxes": GT, "labels": torch.tensor([0, 1])},
        {"boxes": GT[:1] + 5, "labels": torch.tensor([1])},
    ]
    preds = [
        {"boxes": t["boxes"].clone(), "scores": torch.full((len(t["labels"]),), 0.9), "labels": t["labels"]}
        for t in targets
    ]
    return preds, targets


def test_mean_average_precision_perfect() -> None:
    preds, targets = _perfect_case()
    assert m.mean_average_precision(preds, targets) == pytest.approx(1.0)
    assert m.coco_map(preds, targets) == pytest.approx(1.0)


def test_mean_average_precision_per_class_average() -> None:
    """Clase 0 perfecta (AP=1) y clase 1 sin detecciones (AP=0) → mAP = 0.5."""

    preds, targets = _perfect_case()
    for p in preds:
        keep = p["labels"] == 0
        p["boxes"], p["scores"], p["labels"] = p["boxes"][keep], p["scores"][keep], p["labels"][keep]
    assert m.mean_average_precision(preds, targets) == pytest.approx(0.5)


def test_mean_average_precision_ignores_pred_only_classes() -> None:
    """Las clases que solo aparecen en las predicciones no cuentan en la media."""

    preds, targets = _perfect_case()
    preds[0]["boxes"] = torch.cat([preds[0]["boxes"], torch.tensor([[70.0, 70.0, 80.0, 80.0]])])
    preds[0]["scores"] = torch.cat([preds[0]["scores"], torch.tensor([0.1])])
    preds[0]["labels"] = torch.cat([preds[0]["labels"], torch.tensor([7])])
    assert m.mean_average_precision(preds, targets) == pytest.approx(1.0)


def test_coco_map_penalizes_loose_boxes() -> None:
    """Con IoU ≈ 0.68 cuenta a 0.5–0.65 (4 de 10 umbrales) → COCO mAP = 0.4."""

    targets = [{"boxes": GT[:1], "labels": torch.tensor([0])}]
    preds = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 6.8]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    assert m.mean_average_precision(preds, targets, 0.5) == pytest.approx(1.0)
    assert m.coco_map(preds, targets) == pytest.approx(0.4)
