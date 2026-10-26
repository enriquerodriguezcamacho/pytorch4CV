# Chuleta de PyTorch para Computer Vision

Referencia rápida de una página: patrones de código y fórmulas que se repiten en los
exámenes. La explicación completa está en el README de cada tema.

## 1. Patrones de código

```python
# --- Bucle de entrenamiento -------------------------------------------------
model.train()
for x, y in loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

# --- Evaluación ---------------------------------------------------------------
model.eval()
with torch.no_grad():
    preds = model(x).argmax(dim=1)

# --- Gradiente respecto a la ENTRADA (saliency, FGSM) -------------------------
x = x.clone().detach().requires_grad_(True)
loss = F.cross_entropy(model(x), y)
grad = torch.autograd.grad(loss, x)[0]          # o loss.backward(); x.grad

# --- Forward hook (style transfer, Grad-CAM) ----------------------------------
acts = {}
handle = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
model(x); handle.remove()

# --- Congelar y sustituir la cabeza (transfer learning) -----------------------
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)

# --- Guardar / cargar ---------------------------------------------------------
torch.save(model.state_dict(), path)
model.load_state_dict(torch.load(path, weights_only=True))
```

## 2. Tamaños

| Operación | Fórmula |
|---|---|
| Conv / Pool | $\lfloor (H + 2p - d(k-1) - 1)/s \rfloor + 1$ |
| Conv transpuesta | $(H-1)s - 2p + d(k-1) + \text{output\_padding} + 1$ |
| Conv3D (tiempo) | $\lfloor (T + 2p - k)/s \rfloor + 1$ |
| Parámetros de una conv | $C_{out}(C_{in}/g \cdot k_h k_w) + C_{out}$ |
| Parches de ViT | $N = (H/P)(W/P)$; *embedding* $C P^2 \to D$ |

## 3. Fórmulas por tema

**Detección**
- IoU: $\frac{|A\cap B|}{|A|+|B|-|A\cap B|}$
- AP: $\sum_k (R_k - R_{k-1}) P_k$
- YOLO: $\lambda_{coord}\sum(\dots)^2 + \sum(C-\hat C)^2_{obj} + \lambda_{noobj}\sum(C-\hat C)^2_{noobj} + \sum(p-\hat p)^2$, con $\lambda_{coord}=5$, $\lambda_{noobj}=0.5$ y $\sqrt{w}, \sqrt{h}$.

**Segmentación**
- Dice: $\frac{2|A\cap B|}{|A|+|B|}$
- mIoU: media del IoU de cada clase.

**Style transfer**
- Gram: $G = F F^\top / (CHW)$, con $F\in\mathbb R^{C\times HW}$
- $L = \alpha L_C + \beta L_S + \gamma L_{TV}$
- TV: $\frac{1}{2CHW}\sum |x_{i+1,j}-x_{i,j}| + |x_{i,j+1}-x_{i,j}|$

**Profundidad**
- Scale-invariant: $\frac1n\sum d_i^2 - \frac{\lambda}{n^2}(\sum d_i)^2$, con $d_i = \log y_i - \log y_i^*$

**XAI**
- Saliency: $\max_c |\partial S_y/\partial x|$
- IG: $(x-x')\int_0^1 \nabla F(x'+\alpha(x-x'))d\alpha$
- Grad-CAM: $\alpha_k = \frac{1}{Z}\sum_{ij}\frac{\partial y^c}{\partial A^k_{ij}}$, $M = \mathrm{ReLU}(\sum_k \alpha_k A^k)$
- RISE: $S = \frac{1}{\mathbb E[M]\,N}\sum_i f(x\odot M_i)M_i$

**Ataques adversarios**
- FGSM: $x' = x + \varepsilon\,\mathrm{sign}(\nabla_x L)$
- I-FGSM / PGD: $x^{t+1} = \Pi_{x,\varepsilon}\big(x^t \pm \alpha\,\mathrm{sign}(\nabla_x L)\big)$
- Normas: $\|\delta\|_\infty \le \|\delta\|_2 \le \|\delta\|_1$

**VAE**
- $z = \mu + \sigma\odot\epsilon$, con $\sigma = e^{0.5\log\sigma^2}$
- $KL = -\frac12\sum(1+\log\sigma^2-\mu^2-\sigma^2)$
- $-\text{ELBO} = L_{rec} + \beta\,KL$

**GAN**
- $\min_G\max_D\ \mathbb E[\log D(x)] + \mathbb E[\log(1-D(G(z)))]$
- No saturante: $L_G = \mathrm{BCE}(D(G(z)), 1)$
- WGAN-GP: $\lambda(\|\nabla D(\hat x)\|_2 - 1)^2$

**Atención**
- $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$
- Bloque pre-norm: $x \mathrel{+}= \mathrm{MHA}(\mathrm{LN}(x))$, después $x \mathrel{+}= \mathrm{FFN}(\mathrm{LN}(x))$

**Destilación**
- $L = \alpha\,\mathrm{CE}(y, s) + (1-\alpha)\,T^2\,\mathrm{KL}(\mathrm{softmax}(t/T)\,\|\,\mathrm{softmax}(s/T))$

**Audio**
- Mel: $m = 2595\log_{10}(1+f/700)$
- WER: $(S+D+I)/N$

## 4. ¿Qué pérdida uso?

| Tarea | Salida del modelo | Pérdida |
|---|---|---|
| Clasificación multiclase | *logits* `(B, K)` | `nn.CrossEntropyLoss` (opcionalmente `label_smoothing`) |
| Multi-etiqueta / binaria | *logits* `(B, K)` | `nn.BCEWithLogitsLoss` (`pos_weight` si hay desbalanceo) |
| Segmentación semántica | *logits* `(B, K, H, W)` | CE con `ignore_index` (+ Dice) |
| Regresión de cajas | `(B, 4)` | `SmoothL1` / GIoU |
| Profundidad | `(B, 1, H, W)` | *Scale-invariant* / L1 |
| Reconstrucción (AE/VAE) | imagen | MSE / BCE + KL |
| GAN | *logit* de D | `BCEWithLogitsLoss` |
| Destilación | *logits* de alumno y profesor | `KLDivLoss(reduction="batchmean")` × T² |
| *Metric learning* | *embeddings* | `TripletMarginLoss` / InfoNCE |
| OCR / ASR | secuencia de *log-probs* | `nn.CTCLoss` |
