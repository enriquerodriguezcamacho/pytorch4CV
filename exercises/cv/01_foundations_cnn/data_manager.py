"""
CV01 · ``DataManager`` de la práctica 1 — ejercicios.

Mismo diseño que la práctica: un registro de *datasets*, validación en el constructor,
transformaciones compatibles con ImageNet y creación de los ``DataLoader``. El registro
usa ``FakeData`` para no tener que descargar nada; en la práctica contenía CIFAR-10,
Oxford-IIIT Pet y PCAM.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DataManager:
    """
    Gestiona la carga de datos a partir de un registro interno.

    Attributes:
        DATASET_CONFIGS: registro con ``dataset_class``, ``num_classes``,
            ``class_names``, ``train_args`` y ``val_args`` de cada *dataset*.
    """

    DATASET_CONFIGS: dict[str, dict[str, Any]] = {
        "FAKE": {
            "dataset_class": datasets.FakeData,
            "num_classes": 3,
            "class_names": ["a", "b", "c"],
            "train_args": {"size": 8, "image_size": (3, 32, 32), "num_classes": 3},
            "val_args": {"size": 4, "image_size": (3, 32, 32), "num_classes": 3},
        }
    }

    def __init__(self, config: dict[str, dict[str, Any]]) -> None:
        """
        Valida la configuración y guarda la entrada del registro.

        Args:
            config: ``{"data": {"name": ..., "batch_size": ...}}``.

        Raises:
            ValueError: si falta el nombre o no está en el registro.
        """

        self.config = config
        self.train_dataset: Any = None
        self.val_dataset: Any = None
        self.class_names: list[str] = []
        # TODO: valida el nombre y guarda dataset_name, dataset_config y num_classes
        raise NotImplementedError("TODO")

    def _get_transforms(self) -> transforms.Compose:
        """
        Transformaciones compatibles con ImageNet.

        Returns:
            ``Resize((224, 224))``, ``ToTensor`` y ``Normalize`` con las estadísticas
            de ImageNet, en ese orden.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def _build_dataset(self, args_key: str) -> Any:
        """
        Instancia el *dataset* del registro con sus argumentos y las transformaciones.

        Args:
            args_key: ``"train_args"`` o ``"val_args"``.

        Returns:
            *Dataset* instanciado.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def load_data(self) -> None:
        """
        Carga los *datasets* de entrenamiento y validación y rellena ``class_names``.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def get_train_loader(self) -> DataLoader:
        """
        Returns:
            ``DataLoader`` de entrenamiento **barajado**, con el ``batch_size`` de
            ``config["data"]`` (32 por defecto).

        Raises:
            ValueError: si los datos no se han cargado.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def get_val_loader(self) -> DataLoader:
        """
        Returns:
            ``DataLoader`` de validación **sin barajar**, con el ``batch_size`` de
            ``config["data"]`` (32 por defecto).

        Raises:
            ValueError: si los datos no se han cargado.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")
