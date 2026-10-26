"""
CV03 · Style transfer: modelo con hooks para extraer activaciones — soluciones.
"""

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


def build_tiny_vgg() -> nn.Sequential:
    """
    Extractor de features "tipo VGG" diminuto y aleatorio para pruebas (dado).

    Para un resultado real se usaría
    ``torchvision.models.vgg19(weights="IMAGENET1K_V1").features.eval()``.

    Returns:
        ``nn.Sequential`` con 4 ``Conv2d`` (3->8->8->16->16) y ReLU no in-place.
    """

    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(),
    )


class CustomModel(nn.Module):
    """
    Envuelve un modelo y usa forward hooks para guardar las activaciones de todas las
    capas de un tipo dado (por defecto ``nn.Conv2d``), en orden de ejecución.
    """

    def __init__(
        self, model: nn.Module, layer_type: type[nn.Module] = nn.Conv2d
    ) -> None:
        """
        Constructor de la clase.

        Args:
            model: modelo (congelado) del que se extraen las features.
            layer_type: tipo de capa a la que se engancha un hook (``nn.Conv2d`` o
                ``nn.ReLU``, por ejemplo).
        """

        super().__init__()

        self.model = model
        self.layer_type = layer_type
        self.activations: list[torch.Tensor] = []
        self.hooks: list[RemovableHandle] = []

    def _register_hooks(self) -> None:
        """
        Registra un forward hook en cada módulo de tipo ``self.layer_type`` que añade
        su salida a ``self.activations``. Guarda los handles en ``self.hooks``.
        """

        def forward_hook(
            module: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor
        ) -> None:
            """
            Hook que guarda la salida de la capa.

            Args:
                module: módulo del hook.
                inp: entradas del módulo.
                out: salida del módulo.
            """

            # BEGIN SOLUTION
            self.activations.append(out)
            # END SOLUTION

        # BEGIN SOLUTION
        for module in self.model.modules():
            if isinstance(module, self.layer_type):
                self.hooks.append(module.register_forward_hook(forward_hook))
        # END SOLUTION

    def _remove_hooks_and_activations(self) -> None:
        """
        Elimina todos los hooks y las activaciones guardadas (dada).
        """

        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo envuelto (dado).

        Args:
            x: imagen. Dimensions: [1, 3, height, width].

        Returns:
            Salida del modelo envuelto.
        """

        return self.model(x)

    def obtain_activations(
        self, x: torch.Tensor, layers: list[int], use_grad: bool
    ) -> dict[int, torch.Tensor]:
        """
        Obtiene las activaciones de ciertas capas (índices en orden de ejecución entre
        las capas de tipo ``self.layer_type``). Registra los hooks, hace el forward
        (con o sin gradientes), construye el diccionario y elimina hooks y
        activaciones antes de devolver (también si se lanza el ``ValueError``).

        Args:
            x: imagen. Dimensions: [1, 3, height, width].
            layers: índices de las capas cuyas activaciones se quieren.
            use_grad: ``True`` para que las activaciones formen parte del grafo.

        Returns:
            Diccionario ``{capa: activación}``.

        Raises:
            ValueError: si algún índice no corresponde a ninguna capa enganchada.
        """

        # BEGIN SOLUTION
        self._remove_hooks_and_activations()
        self._register_hooks()
        try:
            with torch.set_grad_enabled(use_grad):
                self.model(x)
            n_layers = len(self.activations)
            if any(layer < 0 or layer >= n_layers for layer in layers):
                raise ValueError(f"Índices de capa fuera de rango [0, {n_layers}).")
            result = {layer: self.activations[layer] for layer in layers}
        finally:
            self._remove_hooks_and_activations()
        return result
        # END SOLUTION
