# Prácticas: dónde repasar cada parte

Las cuatro prácticas son una referencia prioritaria para el examen. Esta tabla relaciona
lo que pedía cada una con su ejercicio en el repositorio, para repetirlo desde cero con
tests. Los nombres pueden variar un poco, porque aquí algunas clases se han dividido en
funciones.

## Práctica 1 · *Transfer learning*

| Se pedía | Dónde |
|---|---|
| `DataManager`: `_get_transforms`, `load_data`, `get_train_loader`, `get_val_loader` | `exercises/cv/01_foundations_cnn/data_manager.py` |
| `ModelHandler.load_model`, `replace_last_layer` | `exercises/cv/01_foundations_cnn/models.py` |
| Bucle común: `_train_one_epoch`, `_validate`, `execute` | `exercises/cv/01_foundations_cnn/strategies.py` |
| `LastLayerStrategy`, `FullFineTuningStrategy` | `exercises/cv/01_foundations_cnn/strategies.py` |
| `FeatureExtractionStrategy.extract_features` | `exercises/cv/01_foundations_cnn/strategies.py` (`_extract_features`) |

## Práctica 2 · YOLOv1

| Se pedía | Dónde |
|---|---|
| `RaccoonDataset._generate_label` | `exercises/cv/02_object_detection/yolo.py::encode_targets` · `exercises/fundamentals/08_data_pipeline/data.py` |
| `iou`, `nms` | `exercises/cv/02_object_detection/boxes.py` |
| `to_image_coords` | `exercises/cv/02_object_detection/yolo.py::cells_to_boxes` |
| `mean_average_precision` | `exercises/cv/02_object_detection/metrics.py` |
| `YOLOLoss`: `_coordinates_loss`, `_object_loss`, `_no_object_loss`, `_class_loss` | `exercises/cv/02_object_detection/yolo.py` (`_coord_loss`, `_object_loss`, `_noobject_loss`, `_class_loss`) |
| `YOLO`: `_create_cnn`, `_create_mlp` | `exercises/cv/02_object_detection/yolo.py::YOLOv1Head` |
| `Trainer`: `_make_epoch_train`, `fit` | `exercises/fundamentals/09_training_loop/trainer.py` |

## Práctica 3 · Explicabilidad

| Se pedía | Dónde |
|---|---|
| `saliency_map` | `exercises/cv/05_explainability/gradients.py` |
| `Occlusion` | `exercises/cv/05_explainability/perturbation.py::occlusion_map` |
| `IntegratedGradients` y su *baseline* | `exercises/cv/05_explainability/gradients.py::integrated_gradients` |
| `GuidedBackprop`: *hooks* de *forward* y *backward* | `exercises/cv/05_explainability/gradients.py::GuidedBackprop` |

## Práctica 4 · Ataques adversarios

| Se pedía | Dónde |
|---|---|
| `FastGradientSignMethod` | `exercises/cv/06_adversarial_attacks/white_box.py::fgsm` |
| `OnePixelAttack`: población, *fitness* y evolución | `exercises/cv/06_adversarial_attacks/black_box.py` |
| `cutmix` | `exercises/fundamentals/07_regularization_normalization/regularization.py` |

## Cómo repasarlas

1. Sin mirar tu entrega, resuelve el ejercicio y pasa sus tests.
2. Compara con tu entrega en `Practicas/` para ver las diferencias de enfoque y de
   estilo.
3. Vuelve a escribir de memoria las piezas que más han caído: `Trainer`, *hooks*, I-FGSM.

> El código y los enunciados originales de las prácticas **no** están en este
> repositorio: se quedan en la carpeta `Practicas/` de tu ordenador.
