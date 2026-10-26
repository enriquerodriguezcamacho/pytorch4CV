"""
CV01 · Transfer learning: utilidades sobre modelos — ejercicios.

Funciones para localizar y sustituir la última capa de un modelo de
``torchvision`` (o de una clase propia), congelar el *backbone*, descongelar
capas de forma progresiva y crear grupos de parámetros con *learning rate*
discriminativo.
"""

from __future__ import annotations

from torch import nn


def set_nested_attr(model: nn.Module, name: str, value: nn.Module) -> None:
    """
    Sustituye un submódulo dado su nombre con puntos (p.ej. ``"classifier.6"``).

    Esta función ya está dada.

    Args:
        model: modelo que contiene el submódulo.
        name: nombre completo del submódulo, tal y como aparece en
            ``model.named_modules()``.
        value: nuevo submódulo.
    """

    parent: nn.Module = model
    *path, last = name.split(".")
    for part in path:
        parent = getattr(parent, part)
    setattr(parent, last, value)


def get_final_layer(model: nn.Module) -> tuple[str, nn.Linear]:
    """
    Localiza la última capa lineal (la "cabeza" de clasificación) de un modelo.

    Reglas, en este orden:

    1. Si el modelo tiene un atributo ``fc`` que es ``nn.Linear`` (ResNet,
       GoogLeNet...), se devuelve ese.
    2. Si tiene ``classifier``: si es ``nn.Linear`` (DenseNet) se devuelve; si es
       un ``nn.Sequential`` (VGG, AlexNet, MobileNet...) se devuelve la última
       ``nn.Linear`` que contenga (``"classifier.6"`` en VGG).
    3. En otro caso (clase propia), la última ``nn.Linear`` que aparezca en
       ``model.named_modules()``.

    Args:
        model: modelo de clasificación.

    Returns:
        Tupla ``(nombre, capa)`` con el nombre completo (con puntos) y la capa.

    Raises:
        ValueError: si el modelo no contiene ninguna ``nn.Linear``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def replace_last_layer(model: nn.Module, num_classes: int) -> nn.Module:
    """
    Sustituye la última capa lineal por una nueva con ``num_classes`` salidas.

    La nueva capa conserva ``in_features`` y el uso de *bias* de la original y
    queda entrenable (``requires_grad=True``). El cambio se hace *in place*.

    Args:
        model: modelo de clasificación (``resnet18``, ``vgg11``, ``alexnet``,
            ``densenet121`` o una clase propia).
        num_classes: número de clases del nuevo problema.

    Returns:
        El mismo modelo, con la cabeza sustituida.
    """

    # TODO: usa get_final_layer y set_nested_attr
    raise NotImplementedError("TODO")


def freeze_backbone(model: nn.Module) -> nn.Module:
    """
    Congela todos los parámetros salvo los de la última capa lineal.

    Args:
        model: modelo de clasificación.

    Returns:
        El mismo modelo, con ``requires_grad=False`` en todo el *backbone* y
        ``requires_grad=True`` en la cabeza.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def unfreeze_layers(model: nn.Module, prefixes: list[str]) -> nn.Module:
    """
    Descongela los parámetros cuyo nombre empieza por alguno de los prefijos.

    Es la pieza básica del descongelado progresivo (*gradual unfreezing*): en la
    época 0 solo se entrena la cabeza, luego se añade ``"layer4"``, después
    ``"layer3"``...

    Args:
        model: modelo de clasificación.
        prefixes: prefijos de nombres de parámetros, p.ej. ``["layer4", "fc"]``.

    Returns:
        El mismo modelo. Los demás parámetros no cambian de estado.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def get_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """
    Devuelve los parámetros con ``requires_grad=True`` (los que verá el optimizador).

    Args:
        model: modelo.

    Returns:
        Lista de parámetros entrenables, en el orden de ``model.parameters()``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Cuenta el número de escalares de los parámetros de un modelo.

    Args:
        model: modelo.
        trainable_only: si es ``True`` solo cuenta los entrenables.

    Returns:
        Número total de parámetros (``sum(p.numel())``).
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def discriminative_lr_param_groups(
    model: nn.Module, layer_groups: list[str], base_lr: float, decay: float = 0.1
) -> list[dict]:
    """
    Crea grupos de parámetros con *learning rate* discriminativo.

    ``layer_groups`` va de la capa más superficial a la más profunda (la cabeza
    al final). El último grupo recibe ``base_lr``, el penúltimo
    ``base_lr * decay``, el anterior ``base_lr * decay**2``... Los parámetros que
    no encajan con ningún prefijo (p.ej. ``conv1`` y ``bn1`` de ResNet) forman un
    grupo extra al principio con ``base_lr * decay**len(layer_groups)``.

    Solo se incluyen parámetros entrenables y se omiten los grupos vacíos.

    Args:
        model: modelo.
        layer_groups: prefijos de nombres, p.ej.
            ``["layer1", "layer2", "layer3", "layer4", "fc"]``.
        base_lr: *learning rate* de la cabeza.
        decay: factor multiplicativo por cada grupo hacia la entrada.

    Returns:
        Lista de diccionarios ``{"params": [...], "lr": float}`` lista para
        ``torch.optim.SGD(groups, ...)``. Orden: grupo extra (si existe) y luego
        los de ``layer_groups`` en el mismo orden.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
