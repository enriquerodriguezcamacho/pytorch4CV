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

## 5. Catálogo de métodos

**Explicación completa:** [`guia_tensores.ipynb`](guia_tensores.ipynb) explica cada método, sus argumentos, cuándo usarlo y sus errores típicos, con un ejemplo ejecutable. Esta tabla solo sirve para localizarlos. Para practicarlos:
[`tensor_ops.py`](tensor_ops.py), con `pytest tests/fundamentals/test_f01_tensor_ops.py`.

| Para... | Métodos |
|---|---|
| Crear con un valor | `torch.zeros`, `ones`, `full`, `eye`, `empty` |
| Crear secuencias | `torch.arange` (sin el final), `linspace` (con el final), `logspace` |
| Crear al azar | `torch.rand`, `randn`, `randint`, `randperm`, `bernoulli`, `multinomial` |
| Crear con la misma forma | `torch.zeros_like`, `ones_like`, `full_like`, `rand_like`, `randn_like` |
| Crear con otra forma y el mismo tipo | `x.new_zeros`, `new_ones`, `new_full`, `new_tensor` |
| Convertir | `torch.tensor` (copia), `from_numpy` (comparte), `.numpy()`, `.tolist()`, `.item()` |
| Inspeccionar | `.shape`, `.size(d)`, `.ndim`, `.numel()`, `.dtype`, `.device`, `.requires_grad` |
| Cambiar el tipo | `.float()`, `.long()`, `.bool()`, `.to(torch.uint8)`, `.to(device)` |
| Cambiar la forma | `reshape`, `view`, `flatten`, `permute`, `transpose`, `movedim`, `squeeze`, `unsqueeze` |
| Repetir | `repeat`, `repeat_interleave`, `expand`, `expand_as`, `broadcast_to` |
| Mover datos | `flip`, `rot90`, `roll`, `unfold` |
| Unir y separar | `cat`, `stack`, `chunk`, `split`, `unbind` |
| Seleccionar | `x[...]`, `index_select`, `gather`, `masked_select`, `nonzero`, `where` |
| Escribir | `scatter`, `scatter_add_`, `masked_fill`, `index_fill` |
| Matrices especiales | `tril`, `triu`, `diag`, `meshgrid` |
| Reducir | `sum`, `mean`, `std`, `var`, `prod`, `median`, `max`, `min`, `amax`, `amin`, `argmax`, `argmin`, `norm` |
| Acumular | `cumsum`, `cumprod` |
| Ordenar y contar | `sort`, `argsort`, `topk`, `unique`, `bincount`, `any`, `all` |
| Matemáticas | `clamp`, `abs`, `round`, `floor`, `ceil`, `sqrt`, `exp`, `log`, `sign`, `maximum`, `minimum`, `lerp` |
| Comparar | `eq`, `gt`, `isnan`, `isinf`, `equal`, `allclose` |
| Álgebra | `@`, `matmul`, `bmm`, `dot`, `.T`, `.mT`, `einsum`, `cdist`, `linalg.inv` |
| Memoria | `clone`, `detach`, `contiguous`, métodos con `_` final (en el sitio) |

| ¿Comparte memoria? | Operaciones |
|---|---|
| Sí | `view`, `permute`, `transpose`, `expand`, `squeeze`, `unsqueeze`, `x[0]`, `detach`, `from_numpy` |
| No | `clone`, `repeat`, `cat`, `stack`, `torch.tensor`, indexado con máscara o lista |

### Ejercicios de `tensor_ops.py`

Son 47 ejercicios muy cortos, cada uno con un ejemplo en su *docstring*; la pista del `TODO` indica qué método usar.

| Bloque | Ejercicios |
|---|---|
| A. Creación | `zeros_ones_like`, `constant_image`, `identity_batch`, `even_numbers`, `interpolation_steps`, `gaussian_noise_like`, `random_labels`, `blank_like`, `causal_mask` |
| B. Tipos | `to_uint8_image`, `mean_of_labels` |
| C. Forma | `add_batch_and_channel`, `flatten_per_sample`, `hwc_batch_to_chw`, `remove_singletons`, `tile_image`, `expand_to_batch`, `repeat_each`, `horizontal_flip`, `rotate_90`, `cyclic_shift`, `frame_signal_1d` |
| D. Unir y separar | `stack_images`, `concat_channels`, `xywh_to_xyxy`, `chunk_heads`, `split_channels` |
| E. Indexado | `select_rows`, `replace_negatives`, `fill_masked`, `true_coordinates`, `sort_by_score`, `argmax_2d`, `labels_to_one_hot` |
| F. Reducciones | `channel_mean_std`, `minmax_per_image`, `l2_normalize`, `cosine_similarity_matrix`, `running_sum`, `count_per_class`, `unique_with_counts`, `rows_all_any`, `clip_boxes`, `blend` |
| G. Memoria | `independent_copy`, `to_python_list`, `nearly_equal` |

## Referencias
* [PyTorch docs – Tensors](https://docs.pytorch.org/docs/stable/tensors.html)
* [srush/Tensor-Puzzles](https://github.com/srush/Tensor-Puzzles)
* [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
