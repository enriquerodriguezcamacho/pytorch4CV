# F05 · Funciones de pérdida

> **Objetivo:** saber elegir, programar desde cero y depurar la función de pérdida
> adecuada para cada problema de visión: clasificación, segmentación, detección,
> profundidad, *metric learning*, destilación, generativos y OCR. En los exámenes
> casi siempre hay que programar una pérdida (scale-invariant, Gram/estilo/TV,
> YOLO, GAN condicional, KD con $T^2$...). Este tema es la base de todas ellas.

Fichero a completar: [`losses.py`](losses.py) · Tests: `pytest tests/fundamentals/test_f05_losses.py`

---

## 1. Teoría general

### 1.1 Qué es una pérdida
Una pérdida $\mathcal{L}(\hat y, y)$ es un escalar diferenciable que mide lo mal
que lo hace el modelo. El optimizador minimiza su media empírica:
$\theta^{\star} = \arg\min_\theta \frac{1}{N}\sum_i \ell(f_\theta(x_i), y_i)$.
La mayoría de pérdidas salen de **máxima verosimilitud**: minimizar
$-\log p_\theta(y \mid x)$.

| Hipótesis sobre $p(y\mid x)$ | Pérdida resultante |
|---|---|
| Categórica (softmax) | Cross-entropy / NLL |
| Bernoulli (sigmoide) | BCE |
| Gaussiana con $\sigma$ fija | MSE |
| Laplace | L1 |
| Gaussiana con $\sigma$ predicha | `GaussianNLLLoss` |
| Poisson | `PoissonNLLLoss` |

### 1.2 El parámetro `reduction`
Todas las pérdidas de `torch.nn` aceptan `reduction`:

| Valor | Resultado | Cuándo |
|---|---|---|
| `"mean"` (defecto) | media de los elementos (en CE con `weight`, **media ponderada**: divide por $\sum_i w_{y_i}$) | entrenamiento normal; la escala no depende del *batch size* |
| `"sum"` | suma | cuando quieres normalizar tú (p.ej. por nº de objetos en detección, como YOLOv1) |
| `"none"` | tensor elemento a elemento | máscaras, *hard negative mining*, focal loss, pesos por muestra, ver qué muestras fallan |

¿Por qué importa? (1) Cambia la escala del gradiente y por tanto el *learning rate*
efectivo: con `"sum"` el gradiente crece con el lote. (2) En una pérdida
combinada ($\lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_2$), los $\lambda$
solo tienen sentido si las reducciones son coherentes. (3) Para aplicar una
máscara (píxeles válidos, cajas con objeto) necesitas `"none"` y después
`(loss * mask).sum() / mask.sum()`.

```python
loss = F.mse_loss(pred, target, reduction="none")      # [B, 1, H, W]
loss = (loss * valid).sum() / valid.sum().clamp_min(1)  # media solo de píxeles válidos
```

### 1.3 Logits, probabilidades y log-probabilidades
* **Logits** $z$: salida de la última capa lineal, sin activación.
* `softmax(z)` → probabilidades que suman 1 (clases excluyentes).
* `sigmoid(z)` → probabilidad independiente por salida (multi-etiqueta, binario).
* `log_softmax(z) = z - logsumexp(z)`: estable numéricamente. **Nunca hagas**
  `torch.log(torch.softmax(z))`.

La regla de oro: **las pérdidas "con logits" (`CrossEntropyLoss`,
`BCEWithLogitsLoss`) reciben la salida cruda del modelo**; el modelo no debe
terminar en `Softmax`/`Sigmoid`.

---

## 2. Catálogo de pérdidas de `torch.nn`

Notación: $B$ lote, $C$ clases, $d = \hat y - y$.

### 2.1 Clasificación

#### `nn.CrossEntropyLoss` (la más importante)
$$\ell_i = -\sum_c w_c\, \tilde y_{ic} \log \frac{e^{z_{ic}}}{\sum_k e^{z_{ik}}}$$
Con índices de clase: $\ell_i = -w_{y_i}\log \mathrm{softmax}(z_i)_{y_i}$.

| Argumento | Forma / dtype | Notas |
|---|---|---|
| `input` | `[B, C]` o `[B, C, d1, d2, ...]` **logits** float | segmentación: `[B, C, H, W]` (¡clases en la dim 1!) |
| `target` (índices) | `[B]` o `[B, H, W]`, **`torch.long`** | valores en $[0, C)$ o `ignore_index` |
| `target` (probabilidades) | misma forma que `input`, float | MixUp, CutMix, destilación con etiquetas suaves |
| `weight` | `[C]` | clases desbalanceadas (inversa de la frecuencia, *median frequency balancing* en segmentación) |
| `ignore_index` | int (defecto `-100`) | **255** en Pascal VOC/Cityscapes (bordes/void); padding en secuencias |
| `label_smoothing` | $\varepsilon \in [0,1]$ | $\tilde y = (1-\varepsilon)\,y + \varepsilon/C$; regulariza (Inception-v3, ViT/DeiT usan 0.1) |

Detalles finos (los comprueban los tests):
* Con `weight` y `reduction="mean"` se divide por $\sum_i w_{y_i}$, **no por $B$**.
* Con `label_smoothing`, la pérdida es
  $(1-\varepsilon)\,\mathrm{NLL} + \frac{\varepsilon}{C}\sum_c w_c(-\log p_c)$.
* Gradiente respecto a los logits: $\partial \ell / \partial z = p - y$ (bonito y
  estable; por eso softmax + CE van juntos).

Casos de uso en CV: clasificación de imágenes (ResNet, ViT), segmentación
semántica por píxel (FCN, U-Net, DeepLab), clasificación de las RoIs en Faster
R-CNN, clasificación de acciones en vídeo, PixelCNN (CE de 256 clases por píxel),
cabeza de clase de DETR (con peso bajo para la clase "no-object").

```python
criterion = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.1)
loss = criterion(model(x), mask)        # [B, C, H, W] vs [B, H, W] long
```

#### `nn.NLLLoss`
$\ell_i = -w_{y_i}\, x_{i, y_i}$ donde `input` ya son **log-probabilidades**.
`CrossEntropyLoss(z) == NLLLoss(log_softmax(z))`. Úsala cuando el modelo
termina en `LogSoftmax` (p.ej. al combinar log-probs de varios modelos,
o en modelos que devuelven log-probs como algunos clasificadores bayesianos).

#### `nn.BCELoss` vs `nn.BCEWithLogitsLoss`
$$\ell = -\big[p_w\, y\log\sigma(z) + (1-y)\log(1-\sigma(z))\big]$$

| | `BCELoss` | `BCEWithLogitsLoss` |
|---|---|---|
| Entrada | probabilidades en $[0,1]$ (tras `sigmoid`) | **logits** |
| Estabilidad | `log(0)` → PyTorch recorta a $-100$ | estable (*log-sum-exp trick*) |
| `pos_weight` | no | sí, `[C]`: $p_w = N_{neg}/N_{pos}$ para positivos raros |
| Autocast (AMP) | no permitido | sí |

Formas: `input` y `target` con **la misma forma**, `target` **float** (puede ser
suave). Versión estable:
$\ell = (1-y)\,z + (1 + (p_w-1)\,y)\,\big(\log(1+e^{-|z|}) + \max(-z, 0)\big)$.

Casos de uso: clasificación **multi-etiqueta** (una imagen con "perro" y "playa";
CheXpert), **máscaras binarias** (Mask R-CNN predice una máscara $28\times28$ por
RoI con BCE; segmentación de lesiones), **GAN** (discriminador: real=1, falso=0;
generador no saturante: falso=1), **objectness** en YOLOv3+/RPN, mapas de calor de
*keypoints* (o MSE), VAE con imágenes binarizadas (MNIST).

### 2.2 Regresión

| Pérdida | Fórmula $\ell(d)$ | Propiedades | Casos de uso en CV |
|---|---|---|---|
| `nn.MSELoss` | $d^2$ | penaliza mucho los *outliers*; gradiente $\propto d$ | profundidad (baseline), *heatmaps* de pose (Hourglass), autoencoders, YOLOv1 (cajas con $\sqrt{w}$, $\sqrt{h}$), reconstrucción, *feature distillation*, *content loss* |
| `nn.L1Loss` | $\lvert d\rvert$ | robusta; mediana; gradiente constante | superresolución (EDSR), pix2pix (L1 + GAN, menos borroso que L2), DETR (cajas), profundidad |
| `nn.SmoothL1Loss(beta)` | $\frac{0.5 d^2}{\beta}$ si $\lvert d\rvert<\beta$; $\lvert d\rvert - 0.5\beta$ si no | cuadrática cerca de 0 y lineal lejos; `beta=0` → L1 | **regresión de cajas en Fast/Faster R-CNN** (con `beta=1/9` en Detectron), SSD, *keypoints* |
| `nn.HuberLoss(delta)` | $0.5 d^2$ si $\lvert d\rvert\le\delta$; $\delta(\lvert d\rvert-0.5\delta)$ si no | `huber(δ) = δ · smooth_l1(β=δ)` | regresión robusta, RL (DQN), profundidad con ruido |

Formas: `input` y `target` iguales (broadcasting avisa con *warning*); float.

#### `nn.GaussianNLLLoss`
$\ell = \tfrac12\big(\log\max(\sigma^2, \epsilon) + \frac{d^2}{\max(\sigma^2,\epsilon)}\big)$.
El modelo predice media **y varianza** (`var > 0`, usa `softplus`). Es la
**incertidumbre aleatoria** de Kendall & Gal (2017): profundidad/normales con
incertidumbre por píxel, regresión de cajas con incertidumbre (KL-Loss).

#### `nn.PoissonNLLLoss`
$\ell = e^{\hat y} - y\,\hat y$ (con `log_input=True`). Para **conteos**:
conteo de multitudes, de células, de fotones (imágenes de bajo nivel de luz).

#### `nn.KLDivLoss`
$$\mathrm{KL}(P\,\|\,Q) = \sum_c P_c(\log P_c - \log Q_c)$$
**Trampa:** `input` son **log-probabilidades** de $Q$ (el modelo) y `target`
probabilidades de $P$ (o log-probs con `log_target=True`).
**Usa `reduction="batchmean"`**: `"mean"` divide también por $C$ y no es la KL
matemática.

```python
# Knowledge distillation (Hinton 2015): suavizar con T y multiplicar por T²
kd = F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1), reduction="batchmean") * T**2
```
¿Por qué $T^2$? Los gradientes de la parte suave escalan como $1/T^2$; al
multiplicar por $T^2$ las dos partes de la pérdida mantienen magnitudes
comparables al cambiar $T$. Casos de uso: KD (DeiT, ResNet-50 → ResNet-18),
VAT (KL entre $p(y\mid x)$ y $p(y\mid x+r)$), *self-distillation* (DINO), consistencia
en semi-supervisado.

### 2.3 *Metric learning* y *ranking*

| Pérdida | Fórmula | Entrada / target | Casos de uso |
|---|---|---|---|
| `nn.CosineEmbeddingLoss(margin)` | $1-\cos(x_1,x_2)$ si $y=1$; $\max(0,\cos - m)$ si $y=-1$ | `x1, x2: [B, D]`, `y ∈ {1,-1}: [B]` | verificación de caras/firmas, *image–text matching* |
| `nn.MarginRankingLoss(margin)` | $\max(0, -y(x_1 - x_2) + m)$ | `x1, x2: [B]` puntuaciones, `y ∈ {1,-1}` | ranking de calidad estética, *learning to rank*, recuperación |
| `nn.TripletMarginLoss(margin, p)` | $\max(0, d(a,p) - d(a,n) + m)$ | `anchor, positive, negative: [B, D]` | **FaceNet**, re-identificación de personas (ReID), recuperación de imágenes |
| `nn.TripletMarginWithDistanceLoss` | igual con `distance_function` arbitraria (p.ej. `1 - cos`) | ídem | ReID con similitud coseno, embeddings normalizados |
| `nn.HingeEmbeddingLoss(margin)` | $x$ si $y=1$; $\max(0, m - x)$ si $y=-1$ | `x`: distancias `[B]`, `y ∈ {1,-1}` | redes siamesas (pérdida contrastiva clásica de Hadsell 2006), aprendizaje semisupervisado de embeddings |

Variante "contrastiva" moderna (no está en `torch.nn`): **InfoNCE**, ver §3.

### 2.4 Márgenes y multi-etiqueta

| Pérdida | Fórmula (por muestra) | Entrada / target | Uso |
|---|---|---|---|
| `nn.SoftMarginLoss` | $\frac{1}{n}\sum \log(1 + e^{-y x})$ | `x` cualquier forma, `y ∈ {1,-1}` | clasificación binaria logística con etiquetas ±1 |
| `nn.MultiLabelSoftMarginLoss` | $-\frac{1}{C}\sum_c [y_c\log\sigma(x_c) + (1-y_c)\log\sigma(-x_c)]$ | `x: [B, C]`, `y ∈ {0,1}: [B, C]` | multi-etiqueta (= BCE con logits media por clase) |
| `nn.MultiLabelMarginLoss` | $\sum_{j\in Y}\sum_{i\notin Y}\frac{\max(0, 1-(x_j - x_i))}{C}$ | `x: [B, C]`, `y: [B, C]` **long** con los índices positivos al principio y relleno `-1` | multi-etiqueta con ranking (positivos por encima de negativos) |
| `nn.MultiMarginLoss(p, margin)` | $\frac{1}{C}\sum_{i\ne y}\max(0, m - x_y + x_i)^p$ | `x: [B, C]`, `y: [B]` long | **hinge multiclase (SVM)**: Carlini–Wagner usa una pérdida tipo margen con $\kappa$ |

### 2.5 Secuencias: `nn.CTCLoss`
*Connectionist Temporal Classification* (Graves 2006) alinea una salida de
longitud $T$ con una etiqueta más corta $S$ **sin alineamiento manual**,
sumando sobre todos los caminos que, tras quitar repeticiones y *blanks*, dan la
etiqueta: $\mathcal{L} = -\log\sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_t p(\pi_t \mid x)$.

| Argumento | Forma | Notas |
|---|---|---|
| `log_probs` | `[T, B, C]` **log_softmax** (¡tiempo primero!) | $C$ incluye el *blank* (índice `blank=0`) |
| `targets` | `[B, S]` o concatenados `[ΣS]`, long | sin *blanks* |
| `input_lengths` | `[B]` | $T_b$ |
| `target_lengths` | `[B]` | $S_b \le T_b$ |

Casos de uso: **OCR** (CRNN: CNN → BiLSTM → CTC; matrículas, texto en escena,
manuscritos), **ASR** (DeepSpeech, wav2vec 2.0 fine-tuning; ver capítulo 9),
lectura de labios. Decodificación *greedy*: argmax por paso, colapsar repeticiones
y quitar *blanks*. `zero_infinity=True` evita `inf` si $T$ es demasiado corta.

---

## 3. Pérdidas de CV que **no** están en `torch.nn`

| Pérdida | Fórmula | Dónde se usa |
|---|---|---|
| **Soft Dice** | $1 - \frac{2\sum p g + s}{\sum p + \sum g + s}$ | segmentación médica (V-Net, nnU-Net), clases muy desbalanceadas; se combina con CE |
| **Soft IoU / Jaccard** | $1 - \frac{\sum pg}{\sum p + \sum g - \sum pg}$ | segmentación; Lovász-Softmax optimiza IoU directamente |
| **Tversky** | $\frac{TP}{TP + \alpha FP + \beta FN}$ | Dice generalizado para controlar FP/FN |
| **Focal loss** | $-\alpha_t(1-p_t)^\gamma\log p_t$ | **RetinaNet** (desbalance extremo fondo/objeto, $\gamma=2$, $\alpha=0.25$), FCOS, detectores *one-stage* |
| **GIoU / DIoU / CIoU** | $1 - \mathrm{IoU} + \frac{\lvert C \setminus (A\cup B)\rvert}{\lvert C\rvert}$ | regresión de cajas en YOLOv4+, DETR (L1 + GIoU); da gradiente aunque no haya solape |
| **Perceptual** | $\sum_l \lVert\phi_l(\hat y) - \phi_l(y)\rVert_2^2$ con $\phi_l$ capas de VGG | superresolución (SRGAN), *style transfer* (content loss), LPIPS como métrica |
| **Gram / estilo** | $\sum_l w_l\lVert G(\phi_l(\hat y)) - G(\phi_l(s))\rVert_F^2$, $G = FF^\top/(CHW)$ | *style transfer* de Gatys (examen Final) |
| **Total variation** | $\sum_{i,j}(x_{i+1,j}-x_{i,j})^2 + (x_{i,j+1}-x_{i,j})^2$ | suavizado en *style transfer*, *feature inversion*, *denoising* |
| **Scale-invariant (Eigen 2014)** | $\frac{1}{n}\sum d_i^2 - \frac{\lambda}{n^2}(\sum d_i)^2$, $d_i = \log \hat y_i - \log y_i$ | **profundidad monocular** (examen Inter); con $\lambda=1$ invariante a $\hat y \to c\hat y$ |
| **InfoNCE / NT-Xent** | $-\log\frac{e^{s(z_i,z_i^+)/\tau}}{\sum_j e^{s(z_i,z_j)/\tau}}$ | SimCLR, MoCo, **CLIP** (simétrica imagen↔texto) |
| **KL del VAE** | $-\tfrac12\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$ | VAE, β-VAE, CVAE (más reconstrucción BCE/MSE) |
| **WGAN-GP** | $D(\tilde x) - D(x) + \lambda(\lVert\nabla_{\hat x}D(\hat x)\rVert_2 - 1)^2$ | GAN estables (crítico 1-Lipschitz, $\lambda=10$) |
| **Hinge GAN** | $D$: $\mathbb{E}[\max(0,1-D(x))] + \mathbb{E}[\max(0,1+D(G(z)))]$ | SAGAN, BigGAN, con *spectral norm* |
| **Normales** | $\frac{1}{HW}\sum(1 - \cos(n, \hat n))$ | estimación de normales de superficie |
| **Chamfer** | $\frac{1}{\lvert P\rvert}\sum_p\min_q\lVert p-q\rVert^2 + \ldots$ | nubes de puntos (cap. 4) |

```python
def scale_invariant_loss(pred, target, lam=0.5, eps=1e-8):     # [B, 1, H, W]
    d = torch.log(pred + eps) - torch.log(target + eps)
    n = d[0].numel()
    d = d.flatten(1)
    return ((d**2).sum(1) / n - lam * d.sum(1) ** 2 / n**2).mean()

def vae_kl(mu, logvar):                                          # [B, Z]
    return -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1).mean()

def gradient_penalty(critic, real, fake):
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    grad, = torch.autograd.grad(critic(x_hat).sum(), x_hat, create_graph=True)
    return ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()
```

---

## 4. ¿Cuándo uso cada cosa? (tarea → pérdida)

| Tarea de CV | Salida del modelo | Pérdida recomendada |
|---|---|---|
| Clasificación multiclase (ImageNet) | logits `[B, C]` | `CrossEntropyLoss(label_smoothing=0.1)` |
| Clasificación con clases desbalanceadas | logits | CE con `weight`, o focal loss |
| Multi-etiqueta (atributos, rayos X) | logits `[B, C]` | `BCEWithLogitsLoss(pos_weight=...)` |
| Segmentación semántica | `[B, C, H, W]` | CE con `ignore_index=255` (+ Dice si hay desbalance) |
| Segmentación binaria / médica | `[B, 1, H, W]` | BCE con logits + Dice |
| Instancias (Mask R-CNN) | máscara por RoI | BCE por píxel (+ CE de clase + Smooth L1 de caja) |
| Detección two-stage (Faster R-CNN) | clase + deltas de caja | CE + Smooth L1 |
| Detección one-stage (RetinaNet) | clase por ancla | focal loss + Smooth L1 / GIoU |
| YOLOv1 | tensor $S\times S\times(5B+C)$ | suma de MSE con $\lambda_{coord}=5$, $\lambda_{noobj}=0.5$ |
| YOLOv3+ (objectness) | logits | BCE + CIoU |
| DETR | conjunto de predicciones | emparejamiento húngaro + CE + L1 + GIoU |
| Profundidad monocular | `[B, 1, H, W]` | scale-invariant (Eigen) / L1 / BerHu |
| Superresolución | imagen | L1 + perceptual (+ adversarial) |
| *Style transfer* | imagen optimizada | content (MSE en features) + estilo (Gram) + TV |
| Pose / *keypoints* | *heatmaps* | MSE sobre *heatmaps* gaussianos |
| Verificación de caras / ReID | embeddings | triplet, ArcFace (CE con margen angular) |
| Auto-supervisado / CLIP | embeddings | InfoNCE |
| Knowledge distillation | logits estudiante | $\alpha$ CE + $(1-\alpha)T^2$ KL |
| VAE | $\mu, \log\sigma^2$, reconstrucción | BCE/MSE + KL |
| GAN | logit del discriminador | BCE con logits (no saturante), hinge o WGAN-GP |
| OCR / ASR | `[T, B, C]` log-probs | `CTCLoss` |
| Conteo (multitudes, células) | mapa de densidad / conteo | MSE en densidad, `PoissonNLLLoss` |
| Regresión con incertidumbre | media y varianza | `GaussianNLLLoss` |
| Ataques adversariales C&W | logits | margen $\max(\max_{i\neq t} z_i - z_t, -\kappa)$ |

---

## 5. Errores típicos de examen

1. **`softmax` antes de `CrossEntropyLoss`** → aplica softmax dos veces: la
   pérdida no baja de ~$\log C$ y los gradientes se aplastan. La CE ya incluye
   `log_softmax`.
2. **`target` float en vez de `long`** con índices: `RuntimeError: expected scalar
   type Long`. (Si es float, PyTorch lo interpreta como probabilidades y exige la
   misma forma que `input`.) Usa `target.long()`.
3. **`BCELoss` sin `sigmoid`** (o `BCEWithLogitsLoss` con `sigmoid`): la primera
   falla con valores fuera de $[0,1]$; la segunda entrena mal en silencio.
4. **Formas en segmentación**: `input [B, C, H, W]`, `target [B, H, W]` (sin canal).
   Un `target [B, 1, H, W]` da error → `target.squeeze(1)`. Si tienes
   `[B, H, W, C]` → `permute(0, 3, 1, 2)`.
5. **`KLDivLoss` con probabilidades en `input`** (debe ser `log_softmax`) o con
   `reduction="mean"` (usa `"batchmean"`). Y el orden: `kl_div(log Q, P)` es
   $\mathrm{KL}(P\|Q)$.
6. Olvidar el **$T^2$** en destilación o aplicar la temperatura solo a uno de los
   dos modelos.
7. **`BCEWithLogitsLoss` con `target` long** → error; debe ser float.
8. **Olvidar `.detach()`** en el profesor (KD), en el falso al entrenar $D$ (GAN) o
   en el *target* de la red objetivo.
9. Etiquetas en $[1, C]$ en vez de $[0, C)$ → `IndexError`/`device-side assert`.
10. Promediar una pérdida enmascarada con `.mean()` (cuenta los ceros); divide por
    `mask.sum()`.
11. `CTCLoss` con `log_probs` en `[B, T, C]`: debe ser `[T, B, C]`.
12. `MSELoss` con formas `[B]` y `[B, 1]`: hace *broadcasting* a `[B, B]` y solo
    lanza un *warning*.

---

## 6. Ejercicios

| # | Función / clase | Dificultad / puntos |
|---|---|---|
| 1 | `reduce_loss` (con `ValueError`) | ★ |
| 2 | `cross_entropy` (índices/probabilidades, `weight`, `ignore_index`, `label_smoothing`, `[B,C,H,W]`) | ★★★ |
| 3 | `binary_cross_entropy_with_logits` estable con `pos_weight` | ★★ |
| 4 | `smooth_l1_loss` y `huber_loss` | ★ |
| 5 | `distillation_kl` ($T^2$·KL, `batchmean`) | ★★ |
| 6 | `sigmoid_focal_loss` (= `torchvision.ops.sigmoid_focal_loss`) | ★★ |
| 7 | `dice_loss` multiclase con `ignore_index` | ★★ |
| 8 | `generalized_iou_loss` (= `torchvision.ops.generalized_box_iou_loss`) | ★★ |
| 9 | `triplet_margin_loss` | ★ |
| 10 | `info_nce` (opción simétrica tipo CLIP) | ★★ |
| 11 | `DistillationLoss`: `__init__` 0.25 · `_hard_loss` 0.25 · `_soft_loss` 0.75 · `forward` 0.25 | ★★★ (1.5 puntos) |
| 12 | `CombinedSegmentationLoss`: `__init__` 0.25 · `forward` 0.75 | ★★★ (1 punto) |

<details><summary>Pistas</summary>

* Ej. 2: trabaja siempre en la dimensión 1 (`log_softmax(logits, dim=1)`); así
  sirve igual para `[B, C]` y `[B, C, H, W]`. Para `ignore_index`, sustituye los
  píxeles ignorados por 0 antes del `gather` y multiplícalos después por la máscara
  de válidos. Con `"mean"` divide por `w[target][valid].sum()`.
* Ej. 3: $-\log\sigma(z) = \mathrm{softplus}(-z)$ y $-\log(1-\sigma(z)) = z + \mathrm{softplus}(-z)$.
  `softplus(u) = max(u, 0) + log1p(exp(-|u|))`.
* Ej. 5: `log_softmax` para el estudiante, `softmax` para el profesor; suma en
  clases y media en el lote.
* Ej. 6: calcula la BCE elemento a elemento con `reduction="none"` y multiplícala por
  $(1-p_t)^\gamma$ y por $\alpha_t$.
* Ej. 7: `F.one_hot(target, C).permute(0, 3, 1, 2)`; suma en `dims=(0, 2, 3)` para
  tener un Dice por clase.
* Ej. 10: `F.normalize`, `logits = z1 @ z2.T / τ`, etiquetas `torch.arange(B)`.
* Ej. 12: registra `class_weights` con `register_buffer` para que se mueva a GPU con
  el módulo.
</details>

## Referencias
* [PyTorch docs – Loss functions](https://docs.pytorch.org/docs/stable/nn.html#loss-functions) y [`torch.nn.functional`](https://docs.pytorch.org/docs/stable/nn.functional.html#loss-functions)
* [torchvision.ops (focal, GIoU, DIoU, CIoU)](https://docs.pytorch.org/vision/stable/ops.html)
* Hinton et al., *Distilling the Knowledge in a Neural Network* (2015) · Lin et al., *Focal Loss for Dense Object Detection* (2017)
* Milletari et al., *V-Net* (2016, Dice) · Rezatofighi et al., *Generalized IoU* (2019)
* Eigen et al., *Depth Map Prediction from a Single Image using a Multi-Scale Deep Network* (2014)
* Schroff et al., *FaceNet* (2015) · Chen et al., *SimCLR* (2020) · Radford et al., *CLIP* (2021)
* Gatys et al., *Image Style Transfer Using CNNs* (2016) · Johnson et al., *Perceptual Losses* (2016)
* Gulrajani et al., *Improved Training of Wasserstein GANs* (2017) · Graves et al., *CTC* (2006)
* Szegedy et al., *Rethinking the Inception Architecture* (2016, label smoothing)
* Repos: [JunMa11/SegLossOdyssey](https://github.com/JunMa11/SegLossOdyssey), [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2), [kornia.losses](https://kornia.readthedocs.io/en/latest/losses.html)
