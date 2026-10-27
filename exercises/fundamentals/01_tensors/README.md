# F01 · Tensores

> **Objetivo:** dominar la creación, indexado, *broadcasting* y reordenación de
> tensores, que son la base de todo lo que se programa en el examen (IoU, NMS,
> matrices de Gram, atención, parches de ViT...).

Fichero a completar: [`tensors.py`](tensors.py) · Tests: `pytest tests/fundamentals/test_f01_tensors.py`

---

## 1. Teoría

### 1.1 ¿Qué es un tensor?
Un `torch.Tensor` es un array n‑dimensional con:

| Atributo | Ejemplo | Comentario |
|---|---|---|
| `shape` / `size()` | `(B, C, H, W)` | Convención de imágenes en PyTorch: **canales primero** |
| `dtype` | `torch.float32`, `torch.int64`, `torch.bool` | Las pérdidas de clasificación piden `target` en `int64` (`long`) |
| `device` | `cpu`, `cuda:0`, `mps` | Todos los operandos deben estar en el mismo dispositivo |
| `requires_grad` | `True/False` | Activa el registro en el grafo de autograd (ver F02) |
| `stride()` | `(C*H*W, H*W, W, 1)` | Cuántos elementos saltar en memoria por cada dimensión |

### 1.2 Creación
```python
torch.tensor([[1, 2], [3, 4]])          # copia los datos, infiere dtype
torch.zeros(2, 3); torch.ones_like(x); torch.full((2, 2), 7.)
torch.arange(0, 10, 2); torch.linspace(0, 1, 5)
torch.rand(2, 3)    # U[0, 1)        torch.randn(2, 3)   # N(0, 1)
torch.randint(0, 10, (4,))           torch.eye(3)
torch.from_numpy(arr)  # comparte memoria con NumPy
```

### 1.3 Indexado
* Básico: `x[0]`, `x[:, 1:3]`, `x[..., -1]`, `x[:, None]` (añade eje).
* Booleano: `x[x > 0]` → devuelve un tensor 1D con los elementos seleccionados.
* Avanzado (*fancy*): `x[idx]` con `idx` tensor de enteros. `x[torch.arange(B), labels]`
  selecciona un elemento por fila (típico para coger el *logit* de la clase correcta).
* `torch.gather(x, dim, index)`: generalización del anterior; `out[i][j] = x[i][index[i][j]]` si `dim=1`.
* `torch.scatter(x, dim, index, src)`: operación inversa (escribe en posiciones), sirve para *one-hot*.
* `torch.where(cond, a, b)`, `masked_fill(mask, value)` (máscaras de atención).

### 1.4 Broadcasting
Dos tensores son compatibles si, **alineando desde la derecha**, cada dimensión
es igual o una de ellas vale 1 (o no existe).

```
a: (N, 1, 4)        boxes1[:, None, :]
b:    (M, 4)        boxes2
→  (N, M, 4)        todas las parejas → base del IoU por pares
```

### 1.5 Cambiar la forma
| Operación | Copia memoria | Cuándo usarla |
|---|---|---|
| `view(*shape)` | No (requiere tensor contiguo) | Aplanar tras convoluciones |
| `reshape(*shape)` | Solo si hace falta | Opción segura por defecto |
| `permute(*dims)` / `transpose(a, b)` | No (cambia *strides*) | `(B,H,W,C)` ↔ `(B,C,H,W)` |
| `contiguous()` | Sí | Antes de `view` tras un `permute` |
| `squeeze` / `unsqueeze` | No | Quitar/añadir ejes de tamaño 1 |
| `flatten(start_dim)` | No (si se puede) | `x.flatten(1)` → `(B, C*H*W)` |
| `cat` / `stack` | Sí | `cat` une por un eje existente; `stack` crea uno nuevo |
| `split` / `chunk` | No | Separar `(x, y, w, h)` o cabezas de atención |
| `repeat` / `expand` | Sí / No | `expand` no copia (solo dims de tamaño 1) |

### 1.6 Reducciones y álgebra
`sum, mean, std, var, max, min, argmax, topk, sort, cumsum, norm` aceptan `dim`
y `keepdim=True` (muy útil para normalizar con broadcasting).
Producto matricial: `@` / `torch.matmul` (con *batch*), `torch.bmm`,
`torch.einsum("bij,bjk->bik", a, b)`. Distancias: `torch.cdist`.

### 1.7 `einsum` en una tabla
| Expresión | Significado |
|---|---|
| `"ij,jk->ik"` | producto matricial |
| `"bchw->bc"` | *global sum pooling* |
| `"bci,bdi->bcd"` | matriz de Gram por lote (*style transfer*) |
| `"bhqd,bhkd->bhqk"` | puntuaciones de atención `QKᵀ` |

---

## 2. ¿Cuándo uso cada cosa? (casos de uso en CV)

| Necesidad en CV | Herramienta |
|---|---|
| Pasar imagen PIL/NumPy `(H, W, C)` a PyTorch | `torch.from_numpy(img).permute(2, 0, 1).float() / 255` |
| Normalizar con media/desviación de ImageNet por canal | `(x - mean[:, None, None]) / std[:, None, None]` |
| IoU de todas las cajas contra todas | broadcasting `(N,1,4)` vs `(1,M,4)` |
| Coger la probabilidad de la clase verdadera | `gather` o indexado avanzado |
| One-hot de una máscara de segmentación `(B,H,W)` | `F.one_hot(...).permute(0, 3, 1, 2)` |
| Dividir una imagen en parches (ViT) | `reshape` + `permute` o `unfold` |
| Matriz de Gram | `einsum` o `bmm` |
| Top‑k predicciones | `topk` |

## 3. Errores típicos de examen
* `view` tras `permute` → `RuntimeError: view size is not compatible...` → usa `reshape` o `.contiguous()`.
* Olvidar `keepdim=True` y que el broadcasting falle o, peor, funcione mal.
* Mezclar `int64` con `float32` en divisiones (`/` siempre devuelve float; `//` entera).
* `torch.max(x, dim)` devuelve **(valores, índices)**.

## 4. Ejercicios
| # | Función | Dificultad |
|---|---|---|
| 1 | `to_chw_tensor` | ★ |
| 2 | `normalize_per_channel` | ★ |
| 3 | `pairwise_sq_distances` (sin bucles) | ★★ |
| 4 | `select_by_index` (`gather`) | ★★ |
| 5 | `one_hot_mask` (`scatter`) | ★★ |
| 6 | `image_to_patches` | ★★★ |
| 7 | `batched_gram_matrix` (`einsum`) | ★★ |
| 8 | `topk_accuracy` | ★★ |
| 9 | `masked_mean` (media solo de los válidos) | ★ |
| 10 | `patches_to_image` (inversa de los parches) | ★★★ |
| 11 | `class_histogram` (`scatter_add_`) | ★★ |
| 12 | `grid_coordinates` (`meshgrid`, orden `(x, y)`) | ★ |
| 13 | `mask_to_box` (= `torchvision.ops.masks_to_boxes`) | ★★★ |
| 14 | `batched_index_select` (`gather` con `expand`) | ★★ |

<details><summary>Pistas</summary>

* Ej. 3: ‖a−b‖² = ‖a‖² + ‖b‖² − 2·a·b.
* Ej. 6: `(B,C,H,W) → (B,C,H/p,p,W/p,p) → (B,H/p,W/p,C,p,p) → (B,N,C·p·p)`.
* Ej. 10: deshaz los pasos del 6 en orden inverso; el `permute` inverso es `(0, 3, 1, 4, 2, 5)`.
* Ej. 13: `rows = mask.any(2)`; el primer `True` es `rows.int().argmax(1)` y el último sale de
  hacer lo mismo con `rows.flip(1)`.
* Ej. 14: `indices[..., None].expand(-1, -1, D)` y `gather` en la dimensión 1.
</details>

## Referencias
* [PyTorch docs – Tensors](https://docs.pytorch.org/docs/stable/tensors.html)
* [srush/Tensor-Puzzles](https://github.com/srush/Tensor-Puzzles)
* [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
