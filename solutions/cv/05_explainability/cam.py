"""
CV05 · Explicabilidad: métodos ad-hoc (CAM, Grad-CAM, atención) — soluciones.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.hooks import RemovableHandle

from .utils import normalize_map, select_target, target_scores


def cam(
    features: torch.Tensor, fc_weight: torch.Tensor, target_class: torch.Tensor
) -> torch.Tensor:
    """
    Class Activation Map clásico (Zhou et al., 2016) para arquitecturas
    ``última conv → GAP → Linear``:

    $M_c(i, j) = \\sum_k w^c_k A_k(i, j)$.

    Args:
        features: activaciones de la última conv. Dimensions: [batch, K, h, w].
        fc_weight: pesos de la capa lineal final. Dimensions: [num_classes, K].
        target_class: clase a explicar de cada muestra. Dimensions: [batch].

    Returns:
        Mapa CAM **sin** ReLU ni normalizar. Dimensions: [batch, h, w].
    """

    # BEGIN SOLUTION
    weights = fc_weight[target_class]  # [batch, K]
    return torch.einsum("bk,bkhw->bhw", weights, features)
    # END SOLUTION


class GradCAM:
    """
    Grad-CAM (Selvaraju et al., 2017) sobre una capa convolucional objetivo.

    Con un *forward hook* se guardan las activaciones $A$ de la capa y con un
    *full backward hook* el gradiente $\\partial f_c / \\partial A$.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
        Constructor: registra los dos *hooks* en ``target_layer``.

        Args:
            model: modelo a explicar.
            target_layer: capa (normalmente la última conv) cuyas activaciones se usan.
        """

        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handles: list[RemovableHandle] = []
        # BEGIN SOLUTION
        self.handles.append(target_layer.register_forward_hook(self._save_activation))
        self.handles.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )
        # END SOLUTION

    def _save_activation(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """
        *Forward hook*: guarda la salida de la capa en ``self.activations``.

        Args:
            module: capa objetivo.
            inputs: entradas de la capa.
            output: salida de la capa. Dimensions: [batch, K, h, w].
        """

        # BEGIN SOLUTION
        self.activations = output.detach()
        # END SOLUTION

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """
        *Backward hook*: guarda $\\partial f_c / \\partial A$ en ``self.gradients``.

        Args:
            module: capa objetivo.
            grad_input: gradientes respecto a las entradas de la capa.
            grad_output: gradientes respecto a la salida. Dimensions: [batch, K, h, w].
        """

        # BEGIN SOLUTION
        self.gradients = grad_output[0].detach()
        # END SOLUTION

    def remove_hooks(self) -> None:
        """Elimina los *hooks* registrados. Ya dado."""

        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def generate(
        self, x: torch.Tensor, target_class: int | None = None
    ) -> torch.Tensor:
        """
        Calcula el mapa Grad-CAM:

        $\\alpha^c_k = \\frac{1}{hw}\\sum_{ij} \\partial f_c / \\partial A^k_{ij}$,
        $M_c = \\mathrm{ReLU}(\\sum_k \\alpha^c_k A^k)$, interpolado (bilineal) al
        tamaño de la entrada y normalizado a [0, 1] por muestra.

        Args:
            x: imágenes de entrada. Dimensions: [batch, channels, height, width].
            target_class: clase a explicar; ``None`` usa la clase predicha.

        Returns:
            Mapa Grad-CAM en [0, 1]. Dimensions: [batch, height, width].
        """

        # BEGIN SOLUTION
        self.model.eval()
        x = x.detach().clone().requires_grad_(True)
        logits = self.model(x)
        targets = select_target(logits, target_class)
        self.model.zero_grad(set_to_none=True)
        target_scores(logits, targets).sum().backward()

        alphas = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, K, 1, 1]
        cam_map = F.relu((alphas * self.activations).sum(dim=1, keepdim=True))
        cam_map = F.interpolate(
            cam_map, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return normalize_map(cam_map.squeeze(1))
        # END SOLUTION


def guided_gradcam(gradcam_map: torch.Tensor, guided_grads: torch.Tensor) -> torch.Tensor:
    """
    Guided Grad-CAM: producto elemento a elemento del mapa Grad-CAM ("dónde")
    con Guided Backprop ("qué píxeles"), difundido a todos los canales.

    Args:
        gradcam_map: mapa Grad-CAM. Dimensions: [batch, height, width].
        guided_grads: gradientes de Guided Backprop. Dimensions:
            [batch, channels, height, width].

    Returns:
        Guided Grad-CAM. Dimensions: [batch, channels, height, width].
    """

    # BEGIN SOLUTION
    return guided_grads * gradcam_map.unsqueeze(1)
    # END SOLUTION


def attention_rollout(attentions: list[torch.Tensor]) -> torch.Tensor:
    """
    Attention rollout (Abnar & Zuidema, 2020) para un ViT.

    En cada capa: media sobre las cabezas, se suma la identidad (conexión
    residual) y se re-normalizan las filas para que sumen 1; después se
    multiplican las matrices de todas las capas: $R = \\hat A_L \\cdots \\hat A_1$.

    Args:
        attentions: lista (de la primera a la última capa) de matrices de atención.
            Dimensions de cada una: [batch, heads, tokens, tokens].

    Returns:
        Matriz de rollout (filas que suman 1). Dimensions: [batch, tokens, tokens].
    """

    # BEGIN SOLUTION
    batch, _, tokens, _ = attentions[0].shape
    eye = torch.eye(tokens, device=attentions[0].device)
    rollout = eye.expand(batch, tokens, tokens)
    for attn in attentions:
        a = attn.mean(dim=1) + eye
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout
    return rollout
    # END SOLUTION
