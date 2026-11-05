# Ejercicios de los apuntes: índice completo

Los **119 ejercicios** del final de cada tema de los apuntes (versión del 12/10/2026)
y dónde resolver cada uno. El profesor ha indicado que son importantes y que pueden caer
en el examen.

**Cómo leer las tablas**

* **Código:** fichero de `exercises/cv/<tema>/` donde se programa. Se corrige con los tests del tema.
* **Teoría:** resuelto en [SOLUCIONES_APUNTES.md](SOLUCIONES_APUNTES.md).
* **Práctica:** ejercicio que el apéndice o el enunciado piden hacer con tus propios datos o herramientas.
* ⭐ **Ya ha caído** en una práctica o un examen: prioridad máxima. Véase [PRACTICAS.md](PRACTICAS.md).

> Los enunciados completos están en `Teoria/apuntes.txt`. Busca, por ejemplo, `4.6 Exercises`.

Para corregir todos los ejercicios nuevos:

```bash
pytest tests/cv -k apuntes
```

## Tema 1 · Fundamentos de CNN (`01_foundations_cnn`)

| Ej. | Título | Tipo | Dónde |
|---|---|---|---|
| 1 | Tabla de arquitecturas | Teoría | Soluciones · bloques en `blocks.py` |
| 2 ⭐ | *Fine-tuning* de ResNet-18 en MNIST | Código | `apuntes.py::adapt_resnet_to_grayscale` + `strategies.py` |
| 3 | Aritmética de convoluciones | Código y teoría | `apuntes.py::conv_output_shape` |
| 4 | Gradiente en ResNet | Teoría | Soluciones |
| 5 | Campo receptivo | Código y teoría | `apuntes.py::receptive_field_per_layer` |
| 6 | Estrategia de transferencia | Teoría | Soluciones |
| 7 | *Data augmentation* | Código | `apuntes.py::build_train_transforms` |
| 8 ⭐ | Extracción de características con VGG16 | Código | `apuntes.py::prepare_vgg_feature_extractor` |

## Tema 2 · Detección (`02_object_detection`)

| Ej. | Título | Tipo | Dónde |
|---|---|---|---|
| 1 | Detector de un objeto | Código | `apuntes.py::SingleObjectDetector`, `single_object_loss` |
| 2 | Propuestas *naive* | Código y teoría | `apuntes.py::random_proposals` |
| 3 ⭐ | IoU con dos formatos | Código | `apuntes.py::iou` · `boxes.py::box_iou` |
| 4 ⭐ | NMS | Código | `boxes.py::nms` |
| 5 ⭐ | mAP | Código | `metrics.py` |
| 6 | Anchors | Código y teoría | `rcnn.py::generate_anchors` · `apuntes.py::count_anchors` |
| 7 | RoI pooling | Código | `apuntes.py::roi_pool_region` · `rcnn.py::roi_pool` |
| 8 | R-CNN y Fast R-CNN | Código | `apuntes.py::RCNN`, `FastRCNN` |
| 9 | Variantes con otros modelos | Teoría | Soluciones |
| 10 ⭐ | Pérdida YOLO y raíz cuadrada | Teoría | Soluciones · código en `yolo.py::YOLOLoss` |
| 11 | Una etapa frente a dos | Teoría | Soluciones |

## Tema 3 · Segmentación y *style transfer* (`03_segmentation_style_transfer`)

| Ej. | Título | Tipo | Dónde |
|---|---|---|---|
| 1 | Segmentación con ventana deslizante | Código y teoría | `segmentation/apuntes.py::sliding_window_segmentation` |
| 2 | Forma de la conv. transpuesta | Código y teoría | `segmentation/unet.py::transposed_conv_output_size` |
| 3 | Conv. transpuesta (s=1, p=0) | Código | `segmentation/unet.py::conv_transpose2d_manual` |
| 4 | Conv. transpuesta general | Código | `segmentation/apuntes.py::conv_transpose2d_general` |
| 5 | Demostración: Bed of Nails | Teoría | Soluciones |
| 6 | Demostración: Nearest Neighbor | Teoría | Soluciones |
| 7 | Upsampling tradicional | Código | `segmentation/apuntes.py` (4 funciones) |
| 8 | FCN | Código | `segmentation/apuntes.py::FCNResNet18` |
| 9 ⭐ | U-Net | Código y teoría | `segmentation/unet.py::UNet` |
| 10 | Entrenar segmentación | Código | `segmentation/apuntes.py::train_segmentation_epoch` |
| 11 ⭐ | Matriz de Gram | Código y teoría | `style_transfer/utils.py::gram_matrix` |
| 12 ⭐ | Pérdidas de *style transfer* | Código | `style_transfer/loss.py` |
| 13 ⭐ | Entrenamiento de *style transfer* | Código | `style_transfer/train.py::Trainer` |
| 14 | Experimentos de pérdida | Práctica | Usa tu `Trainer` · tendencias en Soluciones |
| 15 | Comparativa de tareas | Teoría | Soluciones |

## Tema 4 · 3D y vídeo (`04_3d_video`)

| Ej. | Título | Tipo | Dónde |
|---|---|---|---|
| 1 | Aritmética de Conv3D | Código y teoría | `geometry.py::conv3d_output_size` |
| 2 | Conv3D a mano | Código y teoría | `geometry.py::conv3d_manual` |
| 3 | Inflado de kernels | Código y teoría | `video.py::inflate_conv2d` |
| 4 ⭐ | Pérdida invariante a escala | Código y teoría | `geometry.py::scale_invariant_loss` |
| 5 | Distancias de normales | Código y teoría | `geometry.py::surface_normal_loss` |
| 6 | Memoria: vóxeles y nubes de puntos | Teoría | Soluciones |
| 7 | Chamfer | Código y teoría | `geometry.py::chamfer_distance` |
| 8 | F1@δ | Código | `geometry.py::f1_at_delta` |
| 9 | Invarianza a permutaciones | Código y teoría | `geometry.py::PointCloudClassifier` |
| 10 | Modelos base de vídeo | Código y teoría | `video.py` |

## Tema 5 · Explicabilidad (`05_explainability`)

| Ej. | Título | Tipo | Dónde |
|---|---|---|---|
| 1 | Por qué explicar | Teoría | Soluciones |
| 2 | Grad-CAM generaliza CAM | Teoría | Soluciones |
| 3 | Saturación e IG | Teoría | Soluciones |
| 4 | Complejidad de SHAP y Occlusion | Teoría | Soluciones |
| 5 | Mapas de características | Código | `fundamentals/12_hooks_feature_extraction/hooks.py` |
| 6 | Parches de máxima activación | Código | `feature_viz.py::max_activating_indices` |
| 7 | *Gradient ascent* | Código | `feature_viz.py::gradient_ascent_class_visualization` |
| 8 | Inversión de características | Código | `feature_viz.py::feature_inversion` |
| 9 | DeepDream | Código | `feature_viz.py::deep_dream` |
| 10 | Shapley por Monte Carlo | Código | `apuntes.py::monte_carlo_shapley`, `shapley_pixel_map` |
| 11 ⭐ | Occlusion | Código | `perturbation.py::occlusion_map` |
| 12 ⭐ | RISE | Código | `perturbation.py::RISE`, `VectorizedRISE` |
| 13 ⭐ | Saliency | Código | `gradients.py::saliency_map` |
| 14 | Input × Gradient | Código | `gradients.py::input_x_gradient` |
| 15 ⭐ | Guided Backprop | Código | `gradients.py::GuidedBackprop` |
| 16 | DeconvNet | Código | `gradients.py::DeconvNet` |
| 17 | SmoothGrad | Código | `gradients.py::smoothgrad` |
| 18 ⭐ | Integrated Gradients | Código | `gradients.py::integrated_gradients` |
| 19 | CAM | Código | `cam.py::cam` |
| 20 | Grad-CAM | Código | `cam.py::GradCAM` |
| 21 | Guided Grad-CAM | Código | `cam.py::guided_gradcam` |
| 22 | Loyalty e inverse loyalty | Código y teoría | `apuntes.py::loyalty_scores` |
| 23 | ROAR | Código | `apuntes.py::remove_most_important`, `roar_accuracy` |
| 24 | *Sanity checks* | Código | `apuntes.py::cascading_randomization`, `shuffle_labels`, `global_ssim` · `evaluation.py::randomize_layers` |

## Tema 6 · Ataques adversarios (`06_adversarial_attacks`)

| Ej. | Título | Tipo | Dónde |
|---|---|---|---|
| 1 | Modelos lineales | Teoría | Soluciones · `white_box.py::linear_worst_case` |
| 2 | Tipos de ataque | Teoría | Soluciones |
| 3 | FGSM y PGD | Teoría | Soluciones |
| 4 | Variable en C&W | Teoría | Soluciones · `white_box.py::carlini_wagner_l2` |
| 5 ⭐ | FGSM | Código | `white_box.py::fgsm` |
| 6 ⭐ | Clase PGD | Código | `apuntes.py::PGDAttack` |
| 7 | *Boundary Attack* | Código | `apuntes.py::boundary_attack` |
| 8 ⭐ | CutMix | Código | `fundamentals/07_regularization_normalization/regularization.py::cutmix` |
| 9 | *Adversarial training* con PGD | Código | `apuntes.py::pgd_adversarial_training_epoch` |
| 10 | VAT | Código | `defenses.py::vat_loss` |
| 11 | Evaluación de robustez | Código | `apuntes.py::robustness_curve` |

## Tema 7 · IA generativa (`07_generative_models`)

Todos son teóricos: los 11 están en Soluciones. Aun así, conviene programar lo que explican, porque **ya ha caído**:

| Ej. | Título | Código relacionado |
|---|---|---|
| 1 | Factorización autorregresiva | `autoregressive.py::pixelcnn_nll` |
| 2 | Máscaras A y B | `autoregressive.py::build_causal_mask` |
| 3 | Entrenamiento y generación | `autoregressive.py::sample` |
| 4 | Orden RGB | `autoregressive.py` |
| 5 | Autoencoder | `autoencoders.py` |
| 6 | Generación con AE y VAE | `vae.py` |
| 7 | Variantes | `vae.py`, `vq.py`, `autoencoders.py` |
| 8 ⭐ | Entrenar una GAN | `gan.py::gan_train_step` |
| 9 | Colapso de modos | — |
| 10 ⭐ | Inferencia en GAN | `gan.py`, `conditional.py` |
| 11 ⭐ | Generación condicional | `conditional.py` |

## Tema 8 · Atención y Transformers (`08_attention_transformers`)

| Ej. | Título | Tipo | Dónde |
|---|---|---|---|
| 1 | Por qué dividir entre √d | Teoría | Soluciones |
| 2 | Codificación posicional | Teoría | Soluciones |
| 3 | CNN y ViT | Teoría | Soluciones |
| 4 | *Token* CLS | Teoría | Soluciones |
| 5 | Atención por *batch* | Código | `apuntes.py::batched_attention` |
| 6 | *Patch embedding* | Código | `vit.py::PatchEmbedding`, `VisionTransformer.tokens` |
| 7 | *Encoder* ViT | Código | `attention.py::TransformerEncoderBlock` |
| 8 | Pérdida de destilación | Código | `apuntes.py::deit_loss` |
| 9 | DeiT | Teoría | Soluciones |
| 10 | Ventanas de Swin | Código | `apuntes.py::window_partition`, `window_reverse` |
| 11 | Swin Transformer | Teoría | Soluciones |

## Tema 9 · Audio (`09_audio`)

| Ej. | Título | Tipo | Dónde |
|---|---|---|---|
| 1 | Forma de onda y muestreo | Teoría | Soluciones |
| 2 | Ventanas y saltos | Código y teoría | `apuntes.py::duration_to_samples` |
| 3 | Log-Mel | Teoría | Soluciones · `features.py::mel_filterbank` |
| 4 | *Augmentation* segura | Teoría | Soluciones |
| 5 | Espectrogramas e imágenes | Teoría | Soluciones |
| 6 | Salida del modelo | Teoría | Soluciones · `speech.py` |
| 7 | Colapso de CTC | Código y teoría | `apuntes.py::ctc_collapse` |
| 8 | CTC y atención | Teoría | Soluciones |
| 9 | *Greedy* y *beam search* | Teoría | Soluciones · `speech.py::ctc_greedy_decode` |
| 10 | WER | Código y teoría | `apuntes.py::word_error_alignment` |

## Apéndice A · Ingeniería de software

Vale puntos en todas las entregas (herramientas de calidad y revisión del código).

| Ej. | Título | Tipo | Dónde |
|---|---|---|---|
| 1 | Docstrings en tres formatos | Práctica | Soluciones |
| 2 | Anotaciones de tipos | Práctica | Soluciones |
| 3 | Mypy y Black | Práctica | Soluciones |
| 4 | Complexipy, Bandit, Flake8, Pylint y Ruff | Práctica | Soluciones |
| 5 | Complejidad cognitiva | Práctica | Soluciones |
| 6 | Tests con PyTest | Práctica | Soluciones · ejemplos en `tests/` |
| 7 | Entorno con uv | Práctica | Soluciones |
| 8 | Makefile | Práctica | Soluciones |
