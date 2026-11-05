"""
F01 · Métodos de tensores: ejercicios cortos para memorizarlos — ejercicios.

Cada función se resuelve en una o pocas líneas con el método que indica la pista.
Antes de hacerlos puedes probar los métodos en ``guia_tensores.ipynb``.
"""

import torch

# --------------------------------------------------------------------------- #
# A. Creación                                                                 #
# --------------------------------------------------------------------------- #


def zeros_ones_like(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Crea un tensor de ceros y otro de unos con la forma, ``dtype`` y ``device`` de ``x``.

    Args:
        x: tensor de referencia.

    Returns:
        ``(ceros, unos)``.

    Example:
        >>> zeros_ones_like(torch.tensor([1, 2]))
        (tensor([0, 0]), tensor([1, 1]))
    """
    # TODO: usa torch.zeros_like y torch.ones_like
    raise NotImplementedError("TODO")


def constant_image(
    channels: int, height: int, width: int, value: float
) -> torch.Tensor:
    """Crea una imagen ``float32`` en la que todos los píxeles valen ``value``.

    Args:
        channels: número de canales.
        height: alto.
        width: ancho.
        value: valor de relleno.

    Returns:
        Imagen de forma ``(channels, height, width)``.

    Example:
        >>> constant_image(1, 2, 2, 7.0)
        tensor([[[7., 7.],
                 [7., 7.]]])
    """
    # TODO: usa torch.full (indica dtype)
    raise NotImplementedError("TODO")


def identity_batch(batch: int, n: int) -> torch.Tensor:
    """Crea un *batch* de matrices identidad que no comparten memoria entre sí.

    Args:
        batch: número de matrices.
        n: lado de cada matriz.

    Returns:
        Tensor de forma ``(batch, n, n)``.

    Example:
        >>> identity_batch(2, 2).shape
        torch.Size([2, 2, 2])
    """
    # TODO: usa torch.eye y repeat
    raise NotImplementedError("TODO")


def even_numbers(n: int) -> torch.Tensor:
    """Devuelve los ``n`` primeros números pares: ``0, 2, ..., 2(n - 1)``.

    Args:
        n: cantidad de números.

    Returns:
        Tensor entero de forma ``(n,)``.

    Example:
        >>> even_numbers(4)
        tensor([0, 2, 4, 6])
    """
    # TODO: usa torch.arange con paso 2
    raise NotImplementedError("TODO")


def interpolation_steps(start: float, end: float, steps: int) -> torch.Tensor:
    """Devuelve ``steps`` valores equiespaciados, con ``start`` y ``end`` incluidos.

    Args:
        start: primer valor.
        end: último valor.
        steps: número de valores.

    Returns:
        Tensor de forma ``(steps,)``.

    Example:
        >>> interpolation_steps(0.0, 1.0, 3)
        tensor([0.0000, 0.5000, 1.0000])
    """
    # TODO: usa torch.linspace
    raise NotImplementedError("TODO")


def gaussian_noise_like(x: torch.Tensor, std: float) -> torch.Tensor:
    """Genera ruido gaussiano de media 0 y desviación ``std`` con la forma de ``x``.

    Args:
        x: tensor ``float`` de referencia.
        std: desviación típica.

    Returns:
        Ruido con la forma de ``x``.

    Example:
        >>> gaussian_noise_like(torch.zeros(2, 3), 0.1).shape
        torch.Size([2, 3])
    """
    # TODO: usa torch.randn_like
    raise NotImplementedError("TODO")


def random_labels(n: int, num_classes: int) -> torch.Tensor:
    """Genera ``n`` etiquetas enteras aleatorias en ``[0, num_classes)``.

    Args:
        n: número de etiquetas.
        num_classes: número de clases.

    Returns:
        Tensor ``int64`` de forma ``(n,)``.

    Example:
        >>> random_labels(4, 3).shape
        torch.Size([4])
    """
    # TODO: usa torch.randint
    raise NotImplementedError("TODO")


def blank_like(x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Crea ceros con **otra forma**, pero con el ``dtype`` y el ``device`` de ``x``.

    Args:
        x: tensor de referencia.
        shape: forma del resultado.

    Returns:
        Tensor de ceros de forma ``shape``.

    Example:
        >>> blank_like(torch.tensor([1.5]), (1, 2))
        tensor([[0., 0.]])
    """
    # TODO: usa x.new_zeros
    raise NotImplementedError("TODO")


def causal_mask(n: int) -> torch.Tensor:
    """Crea la máscara causal de atención: ``True`` en la diagonal y por debajo.

    Args:
        n: lado de la máscara.

    Returns:
        Máscara ``bool`` de forma ``(n, n)``.

    Example:
        >>> causal_mask(2)
        tensor([[ True, False],
                [ True,  True]])
    """
    # TODO: usa torch.ones con dtype bool y torch.tril
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# B. Tipos de datos                                                           #
# --------------------------------------------------------------------------- #


def to_uint8_image(x: torch.Tensor) -> torch.Tensor:
    """Convierte una imagen ``float`` a ``uint8``: recorta a [0, 1], escala y redondea.

    Args:
        x: imagen ``float``, con valores que pueden salirse de [0, 1].

    Returns:
        Imagen ``uint8`` con valores en [0, 255].

    Example:
        >>> to_uint8_image(torch.tensor([-1.0, 0.5, 2.0]))
        tensor([  0, 128, 255], dtype=torch.uint8)
    """
    # TODO: usa clamp, round y to(torch.uint8)
    raise NotImplementedError("TODO")


def mean_of_labels(labels: torch.Tensor) -> torch.Tensor:
    """Calcula la media de un tensor entero (``mean`` no admite enteros).

    Args:
        labels: tensor ``int64``.

    Returns:
        Media escalar ``float32``.

    Example:
        >>> mean_of_labels(torch.tensor([1, 2, 6]))
        tensor(3.)
    """
    # TODO: convierte con float() antes de mean
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# C. Cambiar la forma                                                         #
# --------------------------------------------------------------------------- #


def add_batch_and_channel(image: torch.Tensor) -> torch.Tensor:
    """Añade dimensiones de *batch* y de canal a una imagen en escala de grises.

    Args:
        image: imagen de forma ``(H, W)``.

    Returns:
        Tensor de forma ``(1, 1, H, W)``.

    Example:
        >>> add_batch_and_channel(torch.zeros(4, 5)).shape
        torch.Size([1, 1, 4, 5])
    """
    # TODO: usa unsqueeze dos veces o image[None, None]
    raise NotImplementedError("TODO")


def flatten_per_sample(x: torch.Tensor) -> torch.Tensor:
    """Aplana todo salvo la dimensión de *batch*.

    Args:
        x: tensor de forma ``(B, ...)``.

    Returns:
        Tensor de forma ``(B, N)``.

    Example:
        >>> flatten_per_sample(torch.zeros(2, 3, 4)).shape
        torch.Size([2, 12])
    """
    # TODO: usa flatten con start_dim=1
    raise NotImplementedError("TODO")


def hwc_batch_to_chw(x: torch.Tensor) -> torch.Tensor:
    """Reordena un *batch* de ``(B, H, W, C)`` a ``(B, C, H, W)``.

    Args:
        x: tensor de forma ``(B, H, W, C)``.

    Returns:
        Tensor de forma ``(B, C, H, W)``.

    Example:
        >>> hwc_batch_to_chw(torch.zeros(2, 4, 5, 3)).shape
        torch.Size([2, 3, 4, 5])
    """
    # TODO: usa permute
    raise NotImplementedError("TODO")


def remove_singletons(x: torch.Tensor) -> torch.Tensor:
    """Elimina todas las dimensiones de tamaño 1.

    Args:
        x: tensor de cualquier forma.

    Returns:
        Tensor sin dimensiones de tamaño 1.

    Example:
        >>> remove_singletons(torch.zeros(1, 3, 1)).shape
        torch.Size([3])
    """
    # TODO: usa squeeze
    raise NotImplementedError("TODO")


def tile_image(x: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Repite una imagen como un mosaico de ``rows × cols``.

    Args:
        x: imagen de forma ``(C, H, W)``.
        rows: repeticiones en vertical.
        cols: repeticiones en horizontal.

    Returns:
        Imagen de forma ``(C, H · rows, W · cols)``.

    Example:
        >>> tile_image(torch.tensor([[[1, 2]]]), 1, 2)
        tensor([[[1, 2, 1, 2]]])
    """
    # TODO: usa repeat
    raise NotImplementedError("TODO")


def expand_to_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
    """Añade una dimensión de *batch* repitiendo ``x`` **sin copiar memoria**.

    Args:
        x: tensor de forma ``(...)``.
        batch: tamaño del *batch*.

    Returns:
        Vista de forma ``(batch, ...)`` que comparte memoria con ``x``.

    Example:
        >>> expand_to_batch(torch.zeros(2, 3), 4).shape
        torch.Size([4, 2, 3])
    """
    # TODO: usa unsqueeze y expand
    raise NotImplementedError("TODO")


def repeat_each(x: torch.Tensor, times: int) -> torch.Tensor:
    """Repite cada fila de forma consecutiva: ``[a, b]`` pasa a ``[a, a, b, b]``.

    Args:
        x: tensor de forma ``(N, ...)``.
        times: repeticiones de cada fila.

    Returns:
        Tensor de forma ``(N · times, ...)``.

    Example:
        >>> repeat_each(torch.tensor([1, 2]), 2)
        tensor([1, 1, 2, 2])
    """
    # TODO: usa repeat_interleave
    raise NotImplementedError("TODO")


def horizontal_flip(x: torch.Tensor) -> torch.Tensor:
    """Voltea imágenes en horizontal.

    Args:
        x: tensor de forma ``(..., H, W)``.

    Returns:
        Tensor volteado, con la misma forma.

    Example:
        >>> horizontal_flip(torch.tensor([[1, 2, 3]]))
        tensor([[3, 2, 1]])
    """
    # TODO: usa flip en la última dimensión
    raise NotImplementedError("TODO")


def rotate_90(x: torch.Tensor) -> torch.Tensor:
    """Rota imágenes 90° en sentido antihorario.

    Args:
        x: tensor de forma ``(..., H, W)``.

    Returns:
        Tensor de forma ``(..., W, H)``.

    Example:
        >>> rotate_90(torch.tensor([[1, 2], [3, 4]]))
        tensor([[2, 4],
                [1, 3]])
    """
    # TODO: usa torch.rot90
    raise NotImplementedError("TODO")


def cyclic_shift(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Desplaza un mapa ``shift`` posiciones hacia arriba y hacia la izquierda, de forma
    cíclica (es lo que hace SW-MSA en Swin).

    Args:
        x: tensor de forma ``(..., H, W)``.
        shift: desplazamiento.

    Returns:
        Tensor desplazado, con la misma forma.

    Example:
        >>> cyclic_shift(torch.tensor([[1, 2], [3, 4]]), 1)
        tensor([[4, 3],
                [2, 1]])
    """
    # TODO: usa torch.roll con desplazamientos negativos
    raise NotImplementedError("TODO")


def frame_signal_1d(signal: torch.Tensor, frame_size: int, hop: int) -> torch.Tensor:
    """Divide una señal en ventanas solapadas (como en audio).

    Args:
        signal: señal de forma ``(L,)``.
        frame_size: tamaño de la ventana.
        hop: salto entre ventanas.

    Returns:
        Ventanas de forma ``(N, frame_size)``.

    Example:
        >>> frame_signal_1d(torch.arange(6), 3, 2)
        tensor([[0, 1, 2],
                [2, 3, 4]])
    """
    # TODO: usa unfold
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# D. Unir y separar                                                           #
# --------------------------------------------------------------------------- #


def stack_images(images: list[torch.Tensor]) -> torch.Tensor:
    """Une una lista de imágenes en un *batch* (crea una dimensión nueva).

    Args:
        images: lista de ``B`` imágenes de forma ``(C, H, W)``.

    Returns:
        Tensor de forma ``(B, C, H, W)``.

    Example:
        >>> stack_images([torch.zeros(3, 4, 5), torch.ones(3, 4, 5)]).shape
        torch.Size([2, 3, 4, 5])
    """
    # TODO: usa torch.stack
    raise NotImplementedError("TODO")


def concat_channels(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Concatena por canales, como una *skip connection* de U-Net.

    Args:
        a: tensor de forma ``(B, C1, H, W)``.
        b: tensor de forma ``(B, C2, H, W)``.

    Returns:
        Tensor de forma ``(B, C1 + C2, H, W)``.

    Example:
        >>> concat_channels(torch.zeros(2, 3, 4, 4), torch.zeros(2, 5, 4, 4)).shape
        torch.Size([2, 8, 4, 4])
    """
    # TODO: usa torch.cat con dim=1
    raise NotImplementedError("TODO")


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convierte cajas ``(x, y, w, h)`` en ``(x, y, x + w, y + h)``.

    Args:
        boxes: cajas de forma ``(N, 4)``.

    Returns:
        Cajas de forma ``(N, 4)``.

    Example:
        >>> xywh_to_xyxy(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        tensor([[1., 2., 4., 6.]])
    """
    # TODO: usa unbind y torch.stack
    raise NotImplementedError("TODO")


def chunk_heads(x: torch.Tensor, heads: int) -> tuple[torch.Tensor, ...]:
    """Reparte la última dimensión en ``heads`` trozos iguales.

    Args:
        x: tensor de forma ``(..., D)``, con ``D`` divisible por ``heads``.
        heads: número de trozos.

    Returns:
        Tupla de ``heads`` tensores de forma ``(..., D / heads)``.

    Example:
        >>> chunk_heads(torch.arange(4), 2)
        (tensor([0, 1]), tensor([2, 3]))
    """
    # TODO: usa chunk
    raise NotImplementedError("TODO")


def split_channels(x: torch.Tensor, sizes: list[int]) -> tuple[torch.Tensor, ...]:
    """Separa la dimensión 1 en trozos de los tamaños indicados.

    Args:
        x: tensor de forma ``(B, C, ...)``.
        sizes: tamaño de cada trozo, que deben sumar ``C``.

    Returns:
        Tupla con un tensor por trozo.

    Example:
        >>> [part.shape[1] for part in split_channels(torch.zeros(2, 6), [1, 2, 3])]
        [1, 2, 3]
    """
    # TODO: usa split
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# E. Indexado y selección                                                     #
# --------------------------------------------------------------------------- #


def select_rows(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Selecciona filas por índice.

    Args:
        x: tensor de forma ``(N, ...)``.
        indices: índices de forma ``(K,)``.

    Returns:
        Tensor de forma ``(K, ...)``.

    Example:
        >>> select_rows(torch.tensor([[1], [2], [3]]), torch.tensor([2, 0]))
        tensor([[3],
                [1]])
    """
    # TODO: usa index_select
    raise NotImplementedError("TODO")


def replace_negatives(x: torch.Tensor, value: float) -> torch.Tensor:
    """Sustituye los valores negativos por ``value``, sin modificar ``x``.

    Args:
        x: tensor ``float``.
        value: valor de sustitución.

    Returns:
        Tensor nuevo con la forma de ``x``.

    Example:
        >>> replace_negatives(torch.tensor([-1.0, 2.0]), 0.0)
        tensor([0., 2.])
    """
    # TODO: usa torch.where
    raise NotImplementedError("TODO")


def fill_masked(x: torch.Tensor, mask: torch.Tensor, value: float) -> torch.Tensor:
    """Rellena con ``value`` las posiciones donde ``mask`` es ``True``, sin modificar ``x``.

    Args:
        x: tensor de entrada.
        mask: máscara ``bool`` difundible a ``x``.
        value: valor de relleno.

    Returns:
        Tensor nuevo con la forma de ``x``.

    Example:
        >>> fill_masked(torch.tensor([1.0, 2.0]), torch.tensor([True, False]), -5.0)
        tensor([-5.,  2.])
    """
    # TODO: usa masked_fill (sin guion bajo)
    raise NotImplementedError("TODO")


def true_coordinates(mask: torch.Tensor) -> torch.Tensor:
    """Devuelve las coordenadas de las posiciones a ``True``.

    Args:
        mask: máscara ``bool`` de forma ``(H, W)``.

    Returns:
        Coordenadas ``(fila, columna)`` de forma ``(K, 2)``.

    Example:
        >>> true_coordinates(torch.tensor([[False, True], [True, False]]))
        tensor([[0, 1],
                [1, 0]])
    """
    # TODO: usa nonzero
    raise NotImplementedError("TODO")


def sort_by_score(boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Ordena las cajas de mayor a menor *score*, como en NMS.

    Args:
        boxes: cajas de forma ``(N, 4)``.
        scores: puntuaciones de forma ``(N,)``.

    Returns:
        Cajas ordenadas de forma ``(N, 4)``.

    Example:
        >>> sort_by_score(torch.tensor([[0.0] * 4, [1.0] * 4]), torch.tensor([0.1, 0.9]))
        tensor([[1., 1., 1., 1.],
                [0., 0., 0., 0.]])
    """
    # TODO: usa argsort con descending=True
    raise NotImplementedError("TODO")


def argmax_2d(x: torch.Tensor) -> tuple[int, int]:
    """Busca la fila y la columna del valor máximo de una matriz.

    Args:
        x: matriz de forma ``(H, W)``.

    Returns:
        ``(fila, columna)``.

    Example:
        >>> argmax_2d(torch.tensor([[0, 1, 2], [3, 9, 4]]))
        (1, 1)
    """
    # TODO: usa argmax sobre el tensor aplanado y divmod
    raise NotImplementedError("TODO")


def labels_to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convierte etiquetas a *one-hot* sin ``F.one_hot``.

    Args:
        labels: etiquetas ``int64`` de forma ``(N,)``.
        num_classes: número de clases.

    Returns:
        Tensor ``float32`` de forma ``(N, num_classes)``.

    Example:
        >>> labels_to_one_hot(torch.tensor([2, 0]), 3)
        tensor([[0., 0., 1.],
                [1., 0., 0.]])
    """
    # TODO: usa torch.zeros y scatter_ en la dimensión 1
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# F. Reducciones y matemáticas                                                #
# --------------------------------------------------------------------------- #


def channel_mean_std(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcula la media y la desviación de cada canal en todo el *batch*.

    Args:
        x: tensor de forma ``(B, C, H, W)``.

    Returns:
        Media y desviación (con ``std`` por defecto), ambas de forma ``(C,)``.

    Example:
        >>> channel_mean_std(torch.ones(1, 2, 2, 2))
        (tensor([1., 1.]), tensor([0., 0.]))
    """
    # TODO: usa mean y std con dim=(0, 2, 3)
    raise NotImplementedError("TODO")


def minmax_per_image(x: torch.Tensor) -> torch.Tensor:
    """Normaliza cada imagen a [0, 1] por separado. Si es constante, el resultado es 0.

    Args:
        x: tensor ``float`` de forma ``(B, C, H, W)``.

    Returns:
        Tensor normalizado con la misma forma.

    Example:
        >>> minmax_per_image(torch.tensor([[[[0.0, 5.0, 10.0]]]]))
        tensor([[[[0.0000, 0.5000, 1.0000]]]])
    """
    # TODO: usa amin y amax con dim=(1, 2, 3), keepdim=True y clamp_min
    raise NotImplementedError("TODO")


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normaliza a norma ℓ2 unitaria a lo largo de ``dim``, sin dividir entre cero.

    Args:
        x: tensor ``float``.
        dim: dimensión de la normalización.

    Returns:
        Tensor normalizado con la misma forma.

    Example:
        >>> l2_normalize(torch.tensor([3.0, 4.0]))
        tensor([0.6000, 0.8000])
    """
    # TODO: usa norm con keepdim=True y clamp_min
    raise NotImplementedError("TODO")


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Calcula la similitud coseno de todas las parejas.

    Args:
        a: tensor de forma ``(N, D)``.
        b: tensor de forma ``(M, D)``.

    Returns:
        Similitudes de forma ``(N, M)``.

    Example:
        >>> cosine_similarity_matrix(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        tensor([[1., 0.]])
    """
    # TODO: reutiliza l2_normalize y usa @ con b.T
    raise NotImplementedError("TODO")


def running_sum(x: torch.Tensor) -> torch.Tensor:
    """Calcula la suma acumulada, como los TP de una curva PR.

    Args:
        x: tensor de forma ``(N,)``.

    Returns:
        Suma acumulada de forma ``(N,)``.

    Example:
        >>> running_sum(torch.tensor([1, 2, 3]))
        tensor([1, 3, 6])
    """
    # TODO: usa cumsum
    raise NotImplementedError("TODO")


def count_per_class(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Cuenta cuántas veces aparece cada clase, incluidas las que no aparecen.

    Args:
        labels: etiquetas ``int64`` de forma ``(N,)``.
        num_classes: número de clases.

    Returns:
        Conteos de forma ``(num_classes,)``.

    Example:
        >>> count_per_class(torch.tensor([0, 0, 2]), 4)
        tensor([2, 0, 1, 0])
    """
    # TODO: usa torch.bincount con minlength
    raise NotImplementedError("TODO")


def unique_with_counts(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Devuelve los valores distintos, ordenados, y sus apariciones.

    Args:
        x: tensor de cualquier forma.

    Returns:
        ``(valores, apariciones)``, ambos de forma ``(K,)``.

    Example:
        >>> unique_with_counts(torch.tensor([3, 1, 3, 3]))
        (tensor([1, 3]), tensor([1, 3]))
    """
    # TODO: usa torch.unique con return_counts=True
    raise NotImplementedError("TODO")


def rows_all_any(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Indica, por fila, si todos los valores son ``True`` y si lo es alguno.

    Args:
        mask: máscara ``bool`` de forma ``(N, M)``.

    Returns:
        ``(todos, alguno)``, ambos de forma ``(N,)``.

    Example:
        >>> rows_all_any(torch.tensor([[True, False]]))
        (tensor([False]), tensor([True]))
    """
    # TODO: usa all y any con dim=1
    raise NotImplementedError("TODO")


def clip_boxes(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Recorta cajas ``xyxy`` a los límites de la imagen.

    Args:
        boxes: cajas de forma ``(N, 4)``.
        height: alto de la imagen.
        width: ancho de la imagen.

    Returns:
        Cajas recortadas de forma ``(N, 4)``.

    Example:
        >>> clip_boxes(torch.tensor([[-1.0, -2.0, 50.0, 60.0]]), 20, 30)
        tensor([[ 0.,  0., 30., 20.]])
    """
    # TODO: usa clamp con min=0 y un tensor como max
    raise NotImplementedError("TODO")


def blend(a: torch.Tensor, b: torch.Tensor, weight: float) -> torch.Tensor:
    """Mezcla dos tensores: ``a + weight · (b - a)``.

    Args:
        a: primer tensor.
        b: segundo tensor, con la misma forma.
        weight: peso de ``b``.

    Returns:
        Mezcla con la misma forma.

    Example:
        >>> blend(torch.tensor([0.0]), torch.tensor([10.0]), 0.25)
        tensor([2.5000])
    """
    # TODO: usa torch.lerp
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# G. Memoria y conversión                                                     #
# --------------------------------------------------------------------------- #


def independent_copy(x: torch.Tensor) -> torch.Tensor:
    """Crea una copia fuera del grafo y con memoria propia.

    Args:
        x: tensor, que puede tener ``requires_grad=True``.

    Returns:
        Copia sin grafo que no comparte memoria con ``x``.

    Example:
        >>> independent_copy(torch.ones(2, requires_grad=True))
        tensor([1., 1.])
    """
    # TODO: usa detach y clone
    raise NotImplementedError("TODO")


def to_python_list(x: torch.Tensor) -> list:
    """Convierte un tensor en listas de Python.

    Args:
        x: tensor de cualquier forma.

    Returns:
        Listas anidadas con los valores.

    Example:
        >>> to_python_list(torch.tensor([[1, 2]]))
        [[1, 2]]
    """
    # TODO: usa tolist
    raise NotImplementedError("TODO")


def nearly_equal(a: torch.Tensor, b: torch.Tensor, tolerance: float) -> bool:
    """Comprueba si dos tensores ``float`` son iguales salvo una tolerancia absoluta.

    Args:
        a: primer tensor.
        b: segundo tensor.
        tolerance: tolerancia absoluta.

    Returns:
        ``True`` si coinciden.

    Example:
        >>> nearly_equal(torch.tensor([1.0]), torch.tensor([1.01]), 0.1)
        True
    """
    # TODO: usa torch.allclose con rtol=0 y atol
    raise NotImplementedError("TODO")
