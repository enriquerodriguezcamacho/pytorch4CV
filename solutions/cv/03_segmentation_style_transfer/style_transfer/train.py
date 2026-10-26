"""
CV03 · Style transfer: optimización de la imagen sintetizada — soluciones.
"""

import os

import torch

from .config import Config, config
from .loss import StyleTransferLoss
from .model import CustomModel, build_tiny_vgg
from .utils import gram_matrix, load_image, save_image


class Trainer:
    """
    Optimiza los PÍXELES de la imagen sintetizada (la red está congelada).
    """

    def __init__(
        self,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        model: CustomModel | None = None,
        cfg: Config = config,
    ) -> None:
        """
        Constructor de la clase (dado).

        Args:
            content_image: imagen de contenido normalizada. Dimensions:
                [1, 3, height, width].
            style_image: imagen de estilo normalizada. Dimensions:
                [1, 3, height', width'].
            model: modelo con hooks; si es ``None`` se usa ``build_tiny_vgg()``.
            cfg: configuración.
        """

        self.cfg = cfg
        if model is None:
            model = CustomModel(build_tiny_vgg())
        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.content_features = self._init_content_features(
            content_image, cfg.loss.layers_content
        )
        self.style_features = self._init_style_features(
            style_image, cfg.loss.layers_style
        )
        self.synthesized_image = self._init_synthesized_image(
            cfg.training.random_synthesized_image,
            cfg.model.size_images,
            content_image,
        )

        self.optimizer = cfg.training.optimizer(
            [self.synthesized_image], lr=cfg.training.lr
        )
        self.criterion = StyleTransferLoss(
            self.content_features,
            self.style_features,
            self.model,
            style_layer_weights=cfg.loss.style_layer_weights,
            tv_squared=cfg.loss.tv_squared,
        )
        self.losses: dict[str, list[float]] = {
            "content_loss": [],
            "style_loss": [],
            "total_variation_loss": [],
            "loss": [],
        }

    @classmethod
    def from_paths(
        cls, path_content: str, path_style: str, cfg: Config = config
    ) -> "Trainer":
        """
        Crea el ``Trainer`` leyendo las imágenes de disco (dado).

        Args:
            path_content: ruta de la imagen de contenido.
            path_style: ruta de la imagen de estilo.
            cfg: configuración.

        Returns:
            Trainer inicializado.
        """

        size = cfg.model.size_images
        return cls(load_image(path_content, size), load_image(path_style, size), cfg=cfg)

    def _init_content_features(
        self, content_image: torch.Tensor, layers_content: list[int]
    ) -> dict[int, torch.Tensor]:
        """
        Calcula (sin gradientes) las features de contenido.

        Args:
            content_image: imagen de contenido. Dimensions: [1, 3, height, width].
            layers_content: capas de contenido.

        Returns:
            Diccionario ``{capa: features}`` con valores [1, c, h, w].
        """

        # BEGIN SOLUTION
        return self.model.obtain_activations(content_image, layers_content, False)
        # END SOLUTION

    def _init_style_features(
        self, style_image: torch.Tensor, layers_style: list[int]
    ) -> dict[int, torch.Tensor]:
        """
        Calcula (sin gradientes) las matrices de Gram de la imagen de estilo.

        Args:
            style_image: imagen de estilo. Dimensions: [1, 3, height, width].
            layers_style: capas de estilo.

        Returns:
            Diccionario ``{capa: Gram}`` con valores [1, c, c].
        """

        # BEGIN SOLUTION
        feats = self.model.obtain_activations(style_image, layers_style, False)
        return {layer: gram_matrix(f) for layer, f in feats.items()}
        # END SOLUTION

    @staticmethod
    def _init_synthesized_image(
        random: bool, size_images: int, content_image: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Inicializa la imagen sintetizada (un tensor hoja con ``requires_grad=True``).

        Args:
            random: ``True`` para ruido ``N(0, 1)`` de forma
                ``[1, 3, size_images, size_images]``; ``False`` para una copia
                (independiente) de la imagen de contenido.
            size_images: alto y ancho (solo si ``random``).
            content_image: imagen de contenido (solo si no ``random``).

        Returns:
            Imagen sintetizada. Dimensions: [1, 3, height, width].

        Raises:
            ValueError: si ``random`` es ``False`` y no se da ``content_image``.
        """

        # BEGIN SOLUTION
        if random:
            return torch.randn(1, 3, size_images, size_images, requires_grad=True)
        if content_image is None:
            raise ValueError("Hace falta la imagen de contenido.")
        return content_image.detach().clone().requires_grad_(True)
        # END SOLUTION

    def _save_losses(
        self,
        content_loss: torch.Tensor,
        style_loss: torch.Tensor,
        total_variation_loss: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """
        Guarda las pérdidas de la época (dado).
        """

        self.losses["content_loss"].append(content_loss.item())
        self.losses["style_loss"].append(style_loss.item())
        self.losses["total_variation_loss"].append(total_variation_loss.item())
        self.losses["loss"].append(loss.item())

    def _train_one_epoch(self) -> None:
        """
        Un paso de optimización de la imagen. Debe funcionar con Adam y con LBFGS, así
        que usa una ``closure`` (``optimizer.step(closure)``) que pone a cero los
        gradientes, calcula las pérdidas, hace ``backward`` y devuelve la total.
        Guarda las pérdidas UNA vez por época (las de la última llamada a la closure).
        """

        # BEGIN SOLUTION
        last: list[tuple[torch.Tensor, ...]] = []

        def closure() -> torch.Tensor:
            self.optimizer.zero_grad()
            losses = self.criterion(self.synthesized_image)
            losses[-1].backward()
            last.append(tuple(v.detach() for v in losses))
            return losses[-1]

        self.optimizer.step(closure)
        self._save_losses(*last[-1])
        # END SOLUTION

    def fit(
        self,
        epochs: int = config.training.epochs,
        save_every: int | None = None,
        save_dir: str | None = None,
    ) -> torch.Tensor:
        """
        Optimiza la imagen durante ``epochs`` épocas (una llamada a
        ``_train_one_epoch`` por época). Si ``save_dir`` no es ``None``, guarda la
        imagen con ``save_image`` en ``f"{save_dir}/{epoch}.png"`` en la época 1, cada
        ``save_every`` épocas y en la última (creando la carpeta si no existe).

        Args:
            epochs: número de épocas.
            save_every: cada cuántas épocas se guarda (por defecto ``epochs // 5``).
            save_dir: carpeta donde guardar las imágenes o ``None``.

        Returns:
            Imagen sintetizada final (desconectada del grafo). Dimensions:
            [1, 3, height, width].
        """

        if save_every is None:
            save_every = max(1, epochs // 5)

        # BEGIN SOLUTION
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        for epoch in range(1, epochs + 1):
            self._train_one_epoch()
            if save_dir is not None and (
                epoch == 1 or epoch % save_every == 0 or epoch == epochs
            ):
                save_image(self.synthesized_image, f"{save_dir}/{epoch}.png")
        return self.synthesized_image.detach().clone()
        # END SOLUTION
