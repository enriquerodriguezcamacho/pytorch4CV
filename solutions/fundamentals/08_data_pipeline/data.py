"""
F08 · Pipeline de datos (Dataset, DataLoader, transforms) — soluciones.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
    random_split,
)

# Estadísticos por canal de ImageNet (RGB), los que esperan los modelos de torchvision.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Ejercicio 1 · Dataset map-style mínimo
# ---------------------------------------------------------------------------
class TensorPairDataset(Dataset):
    """
    Dataset *map-style* que envuelve dos tensores (entradas y etiquetas).

    Equivale a ``torch.utils.data.TensorDataset`` pero con una transformación
    opcional que se aplica solo a la entrada.
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """
        Constructor.

        Args:
            inputs: entradas. Dimensions: [num_samples, ...].
            targets: etiquetas. Dimensions: [num_samples, ...].
            transform: función opcional que se aplica a cada entrada.

        Raises:
            ValueError: si ``inputs`` y ``targets`` no tienen el mismo número de
                muestras.
        """

        super().__init__()

        # BEGIN SOLUTION
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(
                f"inputs ({inputs.shape[0]}) y targets ({targets.shape[0]}) "
                "deben tener el mismo número de muestras."
            )
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        # END SOLUTION

    def __len__(self) -> int:
        """
        Número de muestras del dataset.

        Returns:
            Número de muestras.
        """

        # BEGIN SOLUTION
        return self.inputs.shape[0]
        # END SOLUTION

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Devuelve la muestra ``index``.

        Args:
            index: índice de la muestra.

        Returns:
            Pareja ``(x, y)`` con la transformación aplicada a ``x``.
        """

        # BEGIN SOLUTION
        x, y = self.inputs[index], self.targets[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y
        # END SOLUTION


# ---------------------------------------------------------------------------
# Ejercicio 2 · Dataset de segmentación con transformación conjunta
# ---------------------------------------------------------------------------
class SyntheticSegmentationDataset(Dataset):
    """
    Dataset sintético de segmentación semántica.

    La máscara se deriva de la imagen (``mask = floor(image[0] * num_classes)``),
    así que si a la imagen y a la máscara se les aplica **la misma** transformación
    geométrica la relación se mantiene. Es lo que hay que garantizar en
    segmentación: un flip a la imagen y otro distinto a la máscara rompe las
    etiquetas sin dar ningún error.
    """

    def __init__(
        self,
        num_samples: int = 8,
        image_size: int = 16,
        num_classes: int = 3,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        seed: int = 0,
    ) -> None:
        """
        Constructor (dado).

        Args:
            num_samples: número de muestras.
            image_size: lado de las imágenes cuadradas.
            num_classes: número de clases de la máscara.
            hflip_prob: probabilidad de flip horizontal.
            vflip_prob: probabilidad de flip vertical.
            seed: semilla para generar los datos.
        """

        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(
            num_samples, 3, image_size, image_size, generator=generator
        )
        self.masks = (self.images[:, 0] * num_classes).long().clamp(max=num_classes - 1)
        self.num_classes = num_classes
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

    def __len__(self) -> int:
        """Número de muestras (dado)."""

        return self.images.shape[0]

    def joint_transform(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica la misma transformación geométrica aleatoria a imagen y máscara.

        Flip horizontal con probabilidad ``hflip_prob`` y flip vertical con
        probabilidad ``vflip_prob``. Usa ``torch.rand(1)`` para cada decisión (así
        respeta ``torch.manual_seed``) y decide **una sola vez** para ambos tensores.

        Args:
            image: imagen. Dimensions: [channels, height, width].
            mask: máscara de clases. Dimensions: [height, width].

        Returns:
            Imagen y máscara transformadas, con las mismas dimensiones.
        """

        # BEGIN SOLUTION
        if torch.rand(1).item() < self.hflip_prob:
            image, mask = image.flip(-1), mask.flip(-1)
        if torch.rand(1).item() < self.vflip_prob:
            image, mask = image.flip(-2), mask.flip(-2)
        return image, mask
        # END SOLUTION

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Devuelve la muestra ``index`` transformada.

        Args:
            index: índice de la muestra.

        Returns:
            Imagen ``float32`` y máscara ``int64``. Dimensions: [3, height, width] y
            [height, width].
        """

        # BEGIN SOLUTION
        return self.joint_transform(self.images[index], self.masks[index])
        # END SOLUTION


# ---------------------------------------------------------------------------
# Ejercicio 3 · collate_fn para detección
# ---------------------------------------------------------------------------
def detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """
    ``collate_fn`` para detección: cada imagen tiene un número distinto de cajas.

    El ``default_collate`` intentaría apilar las cajas y fallaría. Aquí se apilan
    solo las imágenes (todas del mismo tamaño) y los *targets* se devuelven como
    lista, que es el formato que esperan los detectores de torchvision.

    Args:
        batch: lista de ``(image, target)``. ``image`` tiene Dimensions:
            [channels, height, width] y ``target`` es un dict con ``"boxes"``
            (Dimensions: [num_boxes_i, 4]) y ``"labels"`` (Dimensions:
            [num_boxes_i]).

    Returns:
        Imágenes apiladas (Dimensions: [batch, channels, height, width]) y lista
        de longitud ``batch`` con los diccionarios de *targets*.
    """

    # BEGIN SOLUTION
    images = torch.stack([image for image, _ in batch], dim=0)
    targets = [target for _, target in batch]
    return images, targets
    # END SOLUTION


# ---------------------------------------------------------------------------
# Ejercicio 4 · media y desviación por canal
# ---------------------------------------------------------------------------
def compute_mean_std(
    loader: Iterable[torch.Tensor | Sequence[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calcula la media y la desviación típica por canal de todo un dataset.

    Recorre el ``loader`` **una sola vez** acumulando, por canal, la suma, la suma de
    cuadrados y el número de píxeles: $\\sigma^2 = E[x^2] - E[x]^2$ (desviación
    poblacional, ``unbiased=False``). Cada elemento del loader puede ser un tensor
    de imágenes o una tupla/lista cuyo primer elemento son las imágenes.

    Args:
        loader: iterable de lotes. Imágenes con Dimensions:
            [batch, channels, height, width].

    Returns:
        Media y desviación típica por canal. Dimensions: [channels] y [channels].
    """

    # BEGIN SOLUTION
    total = sq_total = None
    count = 0
    for batch in loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.double()
        s = images.sum(dim=(0, 2, 3))
        sq = (images**2).sum(dim=(0, 2, 3))
        total = s if total is None else total + s
        sq_total = sq if sq_total is None else sq_total + sq
        count += images.shape[0] * images.shape[2] * images.shape[3]
    if total is None or sq_total is None:
        raise ValueError("El loader está vacío.")
    mean = total / count
    std = (sq_total / count - mean**2).clamp_min(0.0).sqrt()
    return mean.float(), std.float()
    # END SOLUTION


# ---------------------------------------------------------------------------
# Ejercicio 5 · sampler para clases desbalanceadas
# ---------------------------------------------------------------------------
def make_weighted_sampler(
    labels: torch.Tensor,
    num_samples: int | None = None,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    """
    Crea un ``WeightedRandomSampler`` que equilibra las clases.

    El peso de cada muestra es la inversa de la frecuencia de su clase,
    $w_i = 1 / n_{y_i}$, de modo que en media se ven todas las clases por igual.
    Se muestrea **con reemplazo**.

    Args:
        labels: etiqueta entera de cada muestra. Dimensions: [num_samples].
        num_samples: número de índices por época (por defecto ``len(labels)``).
        generator: generador opcional para reproducibilidad.

    Returns:
        Sampler con ``weights`` de Dimensions: [len(labels)].
    """

    # BEGIN SOLUTION
    labels = labels.long()
    counts = torch.bincount(labels).double()
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels) if num_samples is None else num_samples,
        replacement=True,
        generator=generator,
    )
    # END SOLUTION


# ---------------------------------------------------------------------------
# Ejercicio 6 · desnormalizar para visualizar
# ---------------------------------------------------------------------------
def denormalize(
    x: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """
    Deshace ``transforms.Normalize(mean, std)``: $x \\cdot \\sigma + \\mu$ por canal.

    Funciona tanto con una imagen como con un lote (el canal es la dimensión -3).

    Args:
        x: imagen(es) normalizada(s). Dimensions: [channels, height, width] o
            [batch, channels, height, width].
        mean: media de cada canal.
        std: desviación típica de cada canal.

    Returns:
        Tensor desnormalizado con las mismas dimensiones que ``x``.
    """

    # BEGIN SOLUTION
    m = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    s = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    return x * s + m
    # END SOLUTION


# ---------------------------------------------------------------------------
# Ejercicio 7 · split reproducible y DataLoader reproducible
# ---------------------------------------------------------------------------
def train_val_split(
    dataset: Dataset, val_fraction: float = 0.2, seed: int = 42
) -> tuple[Subset, Subset]:
    """
    Divide un dataset en entrenamiento y validación de forma reproducible.

    Usa ``random_split`` con un ``torch.Generator`` propio sembrado con ``seed`` (no
    el global). El tamaño de validación es ``round(len(dataset) * val_fraction)``.

    Args:
        dataset: dataset map-style.
        val_fraction: fracción para validación, en el intervalo abierto (0, 1).
        seed: semilla del generador.

    Returns:
        ``(train_subset, val_subset)``.

    Raises:
        ValueError: si ``val_fraction`` no está en (0, 1).
    """

    # BEGIN SOLUTION
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction debe estar en el intervalo (0, 1).")
    n = len(dataset)  # type: ignore[arg-type]
    n_val = int(round(n * val_fraction))
    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(dataset, [n - n_val, n_val], generator=generator)
    return train, val
    # END SOLUTION


def seed_worker(worker_id: int) -> None:
    """
    ``worker_init_fn`` (dada): siembra NumPy y ``random`` en cada *worker*.

    PyTorch ya siembra ``torch`` en cada worker con ``base_seed + worker_id``,
    pero NumPy y ``random`` no; sin esto todas las copias generarían los mismos
    aumentos aleatorios.

    Args:
        worker_id: identificador del worker (lo pasa el DataLoader).
    """

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    drop_last: bool = False,
    collate_fn: Callable | None = None,
) -> DataLoader:
    """
    Construye un ``DataLoader`` reproducible.

    Debe pasar un ``torch.Generator`` sembrado con ``seed`` (el orden del barajado
    depende solo de él), ``worker_init_fn=seed_worker`` y
    ``pin_memory=torch.cuda.is_available()``.

    Args:
        dataset: dataset map-style.
        batch_size: tamaño del lote.
        shuffle: si se baraja en cada época.
        seed: semilla del generador del DataLoader.
        num_workers: número de procesos de carga.
        drop_last: si se descarta el último lote incompleto.
        collate_fn: función para juntar las muestras (``None`` → la de por defecto).

    Returns:
        DataLoader configurado.
    """

    # BEGIN SOLUTION
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    # END SOLUTION


# ---------------------------------------------------------------------------
# Ejercicio 8 · Dataset estilo YOLOv1 (práctica 2, simplificado a B = 1)
# ---------------------------------------------------------------------------
class YOLODataset(Dataset):
    """
    Dataset que codifica las cajas de cada imagen en la matriz objetivo de YOLOv1.

    Cada anotación es ``[class, x, y, w, h]`` con el centro ``(x, y)`` y el tamaño
    ``(w, h)`` **normalizados a [0, 1]** respecto a la imagen. Con ``B = 1`` la
    matriz objetivo tiene Dimensions: [S, S, C + 5] y, en la celda ``(i, j)``
    (fila, columna) que contiene el centro del objeto:

    * ``target[i, j, :C]`` → *one-hot* de la clase;
    * ``target[i, j, C]`` → 1 (hay objeto);
    * ``target[i, j, C+1:C+5]`` → ``(x_cell, y_cell, w_cell, h_cell)``, con
      ``x_cell = S·x - j``, ``y_cell = S·y - i`` (relativos a la celda, en [0, 1))
      y ``w_cell = S·w``, ``h_cell = S·h`` (en unidades de celda).

    Es el mismo orden que la implementación de la práctica (clases, confianza,
    caja).
    """

    def __init__(
        self,
        images: torch.Tensor,
        annotations: list[torch.Tensor],
        split_size: int = 7,
        num_classes: int = 20,
    ) -> None:
        """
        Constructor (dado).

        Args:
            images: imágenes. Dimensions: [num_samples, 3, height, width].
            annotations: lista (una por imagen) de anotaciones. Cada elemento con
                Dimensions: [num_boxes_i, 5] en formato ``[class, x, y, w, h]``.
            split_size: tamaño ``S`` de la rejilla.
            num_classes: número de clases ``C``.
        """

        super().__init__()
        self.images = images
        self.annotations = annotations
        self.S = split_size
        self.C = num_classes

    def __len__(self) -> int:
        """Número de muestras (dado)."""

        return len(self.annotations)

    def encode_target(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Convierte las anotaciones de una imagen en la matriz objetivo.

        Si dos objetos caen en la misma celda, se queda **el primero** (YOLOv1 solo
        predice un objeto por celda). Un centro en el borde (``x = 1``) va a la
        última celda.

        Args:
            boxes: anotaciones ``[class, x, y, w, h]``. Dimensions: [num_boxes, 5].

        Returns:
            Matriz objetivo. Dimensions: [S, S, C + 5].
        """

        # BEGIN SOLUTION
        S, C = self.S, self.C
        target = torch.zeros(S, S, C + 5)
        for box in boxes:
            cls = int(box[0].item())
            x, y, w, h = (float(v) for v in box[1:].tolist())
            i, j = min(int(S * y), S - 1), min(int(S * x), S - 1)
            if target[i, j, C] == 1:
                continue
            target[i, j, C] = 1.0
            target[i, j, cls] = 1.0
            target[i, j, C + 1 : C + 5] = torch.tensor(
                [S * x - j, S * y - i, S * w, S * h]
            )
        return target
        # END SOLUTION

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Devuelve la imagen y su matriz objetivo.

        Args:
            index: índice de la muestra.

        Returns:
            Imagen y objetivo. Dimensions: [3, height, width] y [S, S, C + 5].
        """

        # BEGIN SOLUTION
        return self.images[index], self.encode_target(self.annotations[index])
        # END SOLUTION


# ---------------------------------------------------------------------------
# Ejercicio 9 · transforms v2 que mueven cajas y máscaras a la vez
# ---------------------------------------------------------------------------
def make_detection_transform(size: int, hflip_prob: float = 0.5) -> Callable:
    """
    Crea una transformación ``torchvision.transforms.v2`` para detección.

    Debe encadenar (``v2.Compose``), en este orden: ``v2.ToImage()``,
    ``v2.RandomHorizontalFlip(p=hflip_prob)``, ``v2.Resize((size, size))`` y
    ``v2.ToDtype(torch.float32, scale=True)``. Gracias a ``tv_tensors``, si se llama
    con ``(image, tv_tensors.BoundingBoxes, tv_tensors.Mask)`` las cajas y la máscara
    se transforman **igual** que la imagen.

    Args:
        size: lado de la imagen de salida.
        hflip_prob: probabilidad del flip horizontal.

    Returns:
        Transformación invocable.
    """

    from torchvision.transforms import v2

    # BEGIN SOLUTION
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=hflip_prob),
            v2.Resize((size, size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    # END SOLUTION
