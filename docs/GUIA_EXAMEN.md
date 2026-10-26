# Guía del examen práctico de Computer Vision II

Cómo se plantean los exámenes prácticos de la asignatura y cómo preparar cada
parte con este repositorio.

## 1. Formato

Cada examen es un repositorio de **GitHub Classroom** con esta estructura:

```
README.md          # objetivo, evaluación, estructura y puntos por función
pyproject.toml     # proyecto uv, torch CPU
Makefile           # install / format / lint / test / clean
src/<tema>/        # config.py (dataclasses), <modulo>.py, train.py | main.py, utils.py
tests/<tema>/      # test_<modulo>.py (visión PARCIAL de los tests)
tests/utils.py     # DummyCNN / DummyModel
images/, weights/, notes/ o docs/
```

* **Solo se modifica `src/`.** Algunas funciones ya vienen hechas porque "no aportan
  mucho a la asignatura" (carga de imágenes, *plots*, guardado).
* Lo que hay que hacer se marca con `# TODO` dentro de funciones y métodos que ya
  tienen firma, tipos y *docstring* (estilo Google, con las dimensiones de los tensores).
* El README indica los **puntos de cada función**, por ejemplo: `_style_loss`: 0.75 puntos.

## 2. Evaluación

| Parte | Examen final / extraordinario | Parcial (sobre 5) | Prácticas |
|---|---|---|---|
| Tests automáticos | 9 | 4.5 | 7.5 |
| Inspección del código | 0.5 | 0.25 | 2 |
| Herramientas de calidad (black, isort, mypy, flake8, ruff, complexipy, pylint) | 0.5 | 0.25 | 0.5 |

> Solo ves **parte de los tests**. El profesor ejecuta la batería completa, así que
> no basta con pasar los visibles: la implementación tiene que ser correcta.

En el extraordinario la calidad se puntuó por herramienta:

| Herramientas | Puntos |
|---|---|
| Pylint 8.5–9.5 | 0.25 |
| Pylint > 9.5 | 0.5 |
| Black + isort + complexipy | 0.25 |
| mypy + flake8 | 0.25 |

## 3. Qué ha caído

| Convocatoria | Ejercicios | Tema en este repo |
|---|---|---|
| Parcial | U-Net para estimar profundidad: `ConvBlock`, `EncoderBlock`, `DecoderBlock`, `UNet`. Pérdida *scale-invariant* y `Trainer` (`_make_epoch_train`, `_make_epoch_test`, `fit`) | [CV03](../exercises/cv/03_segmentation_style_transfer), [CV04](../exercises/cv/04_3d_video), [F09](../exercises/fundamentals/09_training_loop) |
| Final (diciembre) | *Style transfer*: `gram_matrix`, hooks en VGG, pérdidas de contenido/estilo/TV y `Trainer` que optimiza la imagen. I‑FGSM con `_compute_loss`, `_gradient_step`, `get_norm` y `fit` | [CV03](../exercises/cv/03_segmentation_style_transfer), [CV06](../exercises/cv/06_adversarial_attacks), [F12](../exercises/fundamentals/12_hooks_feature_extraction) |
| Extraordinario (junio) | RISE: máscaras naive y normal, versión en bucle y vectorizada. GAN condicional con `nn.Embedding` y `train_step` | [CV05](../exercises/cv/05_explainability), [CV07](../exercises/cv/07_generative_models) |
| Práctica 1 | *Transfer learning*: `replace_last_layer`, estrategias LastLayer / FeatureExtraction / FullFineTuning | [CV01](../exercises/cv/01_introduction_transfer_learning) |
| Práctica 2 | YOLOv1: *dataset* S×S×(5B+C), IoU, NMS, `YOLOLoss`, mAP | [CV02](../exercises/cv/02_object_detection), [F08](../exercises/fundamentals/08_data_pipeline) |
| Práctica 3 | XAI: *saliency*, *occlusion*, *integrated gradients*, *guided backprop* | [CV05](../exercises/cv/05_explainability) |
| Práctica 4 | FGSM, *One Pixel Attack*, CutMix | [CV06](../exercises/cv/06_adversarial_attacks), [F07](../exercises/fundamentals/07_regularization_normalization) |

## 4. Qué comprueban los tests

Conocer los patrones habituales te ahorra errores:

1. **Formas y tipos.** `assert out.shape == (B, C, H, W), "Incorrect shape."`
2. **Rangos.** Salida en `[0, 1]` si termina en sigmoide, `x >= 0` tras una ReLU.
3. **Valores exactos con tensores diminutos.** Por ejemplo, la pérdida de contenido de
   un mapa 2×2 debe valer `0.25`, así que el orden de `mean`/`sum` y las normalizaciones importan.
4. **Escalado lineal con el peso.** Con `alpha / 2` la pérdida debe salir dividida entre 2.
5. **Entrenamiento.** Los parámetros cambian tras una época (`torch.norm(new - old) > 0`),
   `model.training` está en `True` en *train* y en `False` en *test*, y el historial de pérdidas crece.
6. **Gradientes.** `requires_grad` correcto y uso de `detach` (`fake.detach()` en el discriminador).
7. **Determinismo.** La misma semilla da el mismo resultado, otra semilla da otro, y la
   versión vectorizada coincide con la de bucle.
8. ***Mocks*.** `patch.object(trainer, "_train_one_epoch")` para contar llamadas
   y `patch("...predict", return_value=("cat", 0.95))`.
9. **Errores.** `pytest.raises(ValueError)` ante una norma desconocida.

Los tests de este repositorio siguen los mismos patrones.

## 5. Errores típicos

* Gram: no iterar el diccionario de capas, o calcular la Gram dos veces (la de estilo ya se guardó como Gram).
* TV *loss*: usar el cuadrado cuando se pide el valor absoluto (o al revés), o llamar al modelo dentro de la TV.
* Recorrer un `DataLoader` en `_train_one_epoch` de *style transfer*, cuando ahí se optimiza **una sola imagen**.
* `torch.random()` no existe: es `torch.rand` / `torch.randn`.
* I‑FGSM:
  * Signo: `+` para *untargeted* y `−` para *targeted*.
  * Recortar a `[x−ε, x+ε]` **y** a `[0, 1]`.
  * `detach()` y `requires_grad_()` en cada iteración.
* GAN:
  * Olvidar `fake.detach()` al entrenar D.
  * Usar `BCELoss` sobre *logits* (lo correcto es `BCEWithLogitsLoss`).
* VAE: `std = exp(0.5 * log_var)`, no `exp(log_var)`.
* No dejar `model.eval()` / `torch.no_grad()` en la evaluación.
* No eliminar los *hooks* (`handle.remove()`).

## 6. Checklist de 60 segundos antes de entregar

- [ ] `make format` (o `black` + `isort`) y `make lint` sin errores.
- [ ] Todas las funciones con tipos y sin `print` de depuración.
- [ ] `pytest` en verde.
- [ ] Sin `TODO` sin resolver ni código comentado sobrante.
- [ ] `git add`, `git commit` y `git push` (¡comprueba que el *push* ha llegado!).
