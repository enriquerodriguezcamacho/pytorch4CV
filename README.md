# pytorch4CV

Guía práctica de **PyTorch para Visión por Computador** pensada para preparar los
exámenes prácticos de *Computer Vision II*: teoría resumida, ejercicios con el mismo
formato que el examen (firmas, tipos y *docstrings* ya dados, `# TODO` dentro) y
tests automáticos que corrigen tu solución.

```
exercises/   ← aquí trabajas tú (generado a partir de solutions/)
solutions/   ← soluciones de referencia con marcadores BEGIN/END SOLUTION
tests/       ← los mismos tests corrigen exercises/ o solutions/
docs/        ← temario resumido, guía del examen y chuleta
tools/       ← make_exercises.py (genera exercises/ desde solutions/)
```

## Instalación

Con [uv](https://docs.astral.sh/uv/) (instala PyTorch para CPU):

```bash
uv sync --extra dev
```

O con `pip`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib pillow tqdm pytest ruff
```

## Cómo se usa

1. Elige un tema y abre su fichero en `exercises/` (p.ej.
   `exercises/cv/06_adversarial_attacks/white_box.py`).
2. Sustituye cada `raise NotImplementedError("TODO")` por tu implementación.
3. Pasa los tests del tema:

   ```bash
   pytest tests/cv/test_cv06_adversarial.py          # tus ejercicios
   pytest tests/cv/test_cv06_adversarial.py -k fgsm  # solo una parte
   pytest --target solutions                         # las soluciones de referencia
   ```

Atajos del `Makefile`: `make test`, `make test-solutions`, `make check`
(¿`exercises/` está sincronizado?), `make lint` y `make format`.

> Si te atascas, la solución está en la misma ruta dentro de `solutions/`. Los tests
> comprueban formas, valores calculados a mano, equivalencia con PyTorch/torchvision,
> determinismo con semilla y el modo `train`/`eval`, igual que en el examen.

## Temas

### Fundamentos (`exercises/fundamentals/`)

| Tema | Contenido | Tests |
|---|---|---|
| F01 · Tensores | HWC→CHW, normalización por canal, distancias por pares, `gather`/`scatter`, parches, Gram, máscaras, caja de una máscara | `test_f01_tensors.py` |
| F02 · Autograd | gradientes, `torch.autograd.Function`, hooks de gradiente, FGSM, GP, acumulación | `test_f02_autograd.py` |
| F03 · `nn.Module` | capas propias, *buffers*, inicialización, congelar capas, MC Dropout, guardar/cargar pesos | `test_f03_nn_module.py` |
| F04 · Activaciones | ReLU, GELU, SiLU, Mish, softmax/log-softmax estables, PReLU... desde cero | `test_f04_activations.py` |
| F05 · Pérdidas | CE, BCE, focal, Dice, GIoU, triplet, InfoNCE, destilación | `test_f05_losses.py` |
| F06 · Optimizadores | SGD con momentum/Nesterov, Adam desde cero, *param groups*, *warmup* + coseno, EMA, L-BFGS | `test_f06_optimizers.py` |
| F07 · Regularización y normalización | Dropout, BatchNorm/LayerNorm/GroupNorm, L2, *label smoothing*, Mixup, CutMix, *early stopping* | `test_f07_regularization.py` |
| F08 · Datos | `Dataset`, `DataLoader`, *collate* de detección, *samplers*, dataset YOLO | `test_f08_data.py` |
| F09 · Bucle de entrenamiento | `Trainer` (`_make_epoch_train`, `_make_epoch_test`, `fit`), *checkpoints*, métricas | `test_f09_training.py` |
| F10 · Capas CNN | tamaños de salida, conv naive/im2col, *pooling*, conv transpuesta, *pixel shuffle* | `test_f10_cnn.py` |
| F11 · Arquitecturas | bloques VGG/residual/Inception/dense/SE, LeNet-5, LSTM, Transformer, autoencoder, siamesa | `test_f11_*.py` |
| F12 · Hooks y features | *forward/backward hooks*, Grad-CAM, Guided Backprop, extracción de features con `torch.fx` | `test_f12_hooks.py` |
| F13 · Aumento de datos | volteo/recorte/redimensionado coherente de imagen, máscara y cajas, brillo/contraste, *cutout*, TTA | `test_f13_augmentation.py` |
| F14 · Inferencia y despliegue | `predict` con `inference_mode`, TorchScript, despliegues *shadow*, A/B (test z) y *canary* | `test_f14_inference_deployment.py` |

### Visión por computador (`exercises/cv/`)

| Tema | Contenido | Ha caído en |
|---|---|---|
| CV01 · Transfer learning | `replace_last_layer`, LastLayer / FeatureExtraction / FullFineTuning, bloques | Práctica 1 |
| CV02 · Detección | IoU, NMS, mAP/COCO mAP, anchors, RoI pooling, RPN, YOLOv1 completo | Práctica 2 |
| CV03 · Segmentación y *style transfer* | conv transpuesta, U-Net, Dice/mIoU, Gram, hooks en VGG, `Trainer` de imagen | Parcial, Final |
| CV04 · 3D y vídeo | Conv3d a mano, pérdida *scale-invariant*, normales, Chamfer, F1@δ, PointNet, modelos de vídeo, I3D | Parcial |
| CV05 · Explicabilidad | saliency, IG, Guided Backprop, Grad-CAM, Occlusion, Shapley, **RISE**, deletion/insertion | Extraordinario, Práctica 3 |
| CV06 · Ataques adversarios | FGSM, PGD ℓ∞/ℓ2, **I-FGSM**, C&W, One Pixel Attack, adversarial training, VAT | Final, Práctica 4 |
| CV07 · Modelos generativos | AE, VAE/β-VAE/CVAE, GAN/WGAN-GP, **GAN condicional**, PixelCNN, VQ, BNN y MC Dropout | Extraordinario |
| CV08 · Atención y Transformers | Nadaraya-Watson, atención, MHA, PE sinusoidal, encoder Pre/Post-LN, **ViT**, *hard distillation* | — |
| CV09 · Audio | ZCR, RMS, mel, DFT/STFT, dB, SpecAugment, CNN sobre mel, CRNN + CTC, WER | — |

## Documentación

* [`docs/GUIA_EXAMEN.md`](docs/GUIA_EXAMEN.md): formato del examen, evaluación, qué ha
  caído y qué comprueban los tests.
* [`docs/TEMARIO.md`](docs/TEMARIO.md): resumen de los apuntes capítulo a capítulo.
* [`docs/CHULETA.md`](docs/CHULETA.md): chuleta de PyTorch para el examen.
* [`CONTRIBUTING.md`](CONTRIBUTING.md): cómo añadir o corregir ejercicios.

## Licencia

[MIT](LICENSE).
