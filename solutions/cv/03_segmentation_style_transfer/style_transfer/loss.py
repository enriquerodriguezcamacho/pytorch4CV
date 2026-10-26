"""
CV03 · Style transfer: pérdida de Gatys (contenido + estilo + variación total) — soluciones.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .config import config
from .model import CustomModel
from .utils import gram_matrix


class StyleTransferLoss(nn.Module):
    """
    Pérdida ``alpha * L_content + beta * L_style + gamma * L_TV``.
    """

    def __init__(
        self,
        content_features: dict[int, torch.Tensor],
        style_features: dict[int, torch.Tensor],
        model: CustomModel,
        style_layer_weights: dict[int, float] | None = None,
        tv_squared: bool = False,
    ) -> None:
        """
        Constructor de la clase (dado).

        Args:
            content_features: mapa de features de la imagen de contenido en cada capa
                de contenido. Dimensions de cada valor: [1, c, h, w].
            style_features: matriz de Gram (normalizada por c·h·w) de la imagen de
                estilo en cada capa de estilo. Dimensions de cada valor: [1, c, c].
            model: modelo con hooks.
            style_layer_weights: peso de cada capa de estilo. Si es ``None`` se usa
                ``1 / n_capas`` para todas (media).
            tv_squared: si es ``True``, la TV usa diferencias al cuadrado en vez de
                valores absolutos.
        """

        super().__init__()

        self.content_features = content_features
        self.style_features = style_features
        self.layers_content = list(content_features.keys())
        self.layers_style = list(style_features.keys())
        self.model = model
        if style_layer_weights is None:
            n = len(self.layers_style)
            style_layer_weights = {layer: 1.0 / n for layer in self.layers_style}
        self.style_layer_weights = style_layer_weights
        self.tv_squared = tv_squared

    def _content_loss(
        self,
        synthesized_features: dict[int, torch.Tensor],
        alpha: float = config.loss.alpha,
    ) -> torch.Tensor:
        """
        Content loss: media, sobre las capas de contenido, del MSE (media sobre todos
        los elementos, es decir, dividido por c·h·w) entre las features de la imagen
        sintetizada y las de contenido.

        Args:
            synthesized_features: features de la imagen sintetizada (puede contener
                más capas de las necesarias). Dimensions de cada valor: [1, c, h, w].
            alpha: peso de la pérdida.

        Returns:
            ``alpha * L_content`` (escalar).
        """

        # BEGIN SOLUTION
        loss = sum(
            F.mse_loss(synthesized_features[layer], self.content_features[layer])
            for layer in self.layers_content
        ) / len(self.layers_content)
        return alpha * loss
        # END SOLUTION

    def _style_loss(
        self,
        synthesized_features: dict[int, torch.Tensor],
        beta: float = config.loss.beta,
    ) -> torch.Tensor:
        """
        Style loss: ``sum_l w_l * MSE(Gram(F_l(x)), G_l^S)`` con los pesos
        ``self.style_layer_weights``. OJO: aquí llegan los mapas de features, no las
        Gram; tienes que calcularlas con ``gram_matrix``.

        Args:
            synthesized_features: features de la imagen sintetizada. Dimensions de
                cada valor: [1, c, h, w].
            beta: peso de la pérdida.

        Returns:
            ``beta * L_style`` (escalar).
        """

        # BEGIN SOLUTION
        loss = sum(
            self.style_layer_weights[layer]
            * F.mse_loss(
                gram_matrix(synthesized_features[layer]), self.style_features[layer]
            )
            for layer in self.layers_style
        )
        return beta * loss
        # END SOLUTION

    def _total_variation_loss(
        self, synthesized_image: torch.Tensor, gamma: float = config.loss.gamma
    ) -> torch.Tensor:
        """
        Total variation loss::

            L_TV = 1 / (2·C·H·W) * sum(|x[i, j+1] - x[i, j]| + |x[i+1, j] - x[i, j]|)

        (con ``tv_squared=True`` se usan diferencias al cuadrado).

        Args:
            synthesized_image: imagen sintetizada. Dimensions: [1, 3, height, width].
            gamma: peso de la pérdida.

        Returns:
            ``gamma * L_TV`` (escalar).
        """

        # BEGIN SOLUTION
        _, c, h, w = synthesized_image.shape
        dh = synthesized_image[:, :, 1:, :] - synthesized_image[:, :, :-1, :]
        dw = synthesized_image[:, :, :, 1:] - synthesized_image[:, :, :, :-1]
        if self.tv_squared:
            total = dh.pow(2).sum() + dw.pow(2).sum()
        else:
            total = dh.abs().sum() + dw.abs().sum()
        return gamma * total / (2 * c * h * w)
        # END SOLUTION

    def forward(
        self, synthesized_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: extrae (CON gradientes) las features de la imagen sintetizada en
        todas las capas de contenido y estilo y calcula las tres pérdidas con los
        pesos por defecto.

        Args:
            synthesized_image: imagen sintetizada. Dimensions: [1, 3, height, width].

        Returns:
            Content loss, style loss, TV loss y pérdida total.
        """

        # BEGIN SOLUTION
        layers = sorted(set(self.layers_content) | set(self.layers_style))
        feats = self.model.obtain_activations(synthesized_image, layers, use_grad=True)
        content = self._content_loss(feats)
        style = self._style_loss(feats)
        tv = self._total_variation_loss(synthesized_image)
        return content, style, tv, content + style + tv
        # END SOLUTION
