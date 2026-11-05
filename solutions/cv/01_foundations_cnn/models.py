"""
CV01 · Transfer learning: utilidades sobre modelos — soluciones.

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

    # BEGIN SOLUTION
    fc = getattr(model, "fc", None)
    if isinstance(fc, nn.Linear):
        return "fc", fc

    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Linear):
        return "classifier", classifier
    if isinstance(classifier, nn.Sequential):
        for idx in reversed(range(len(classifier))):
            if isinstance(classifier[idx], nn.Linear):
                return f"classifier.{idx}", classifier[idx]

    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("El modelo no tiene ninguna capa nn.Linear.")
    return linears[-1]
    # END SOLUTION


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

    # BEGIN SOLUTION usa get_final_layer y set_nested_attr
    name, old = get_final_layer(model)
    new = nn.Linear(old.in_features, num_classes, bias=old.bias is not None)
    new = new.to(device=old.weight.device, dtype=old.weight.dtype)
    set_nested_attr(model, name, new)
    return model
    # END SOLUTION


def freeze_backbone(model: nn.Module) -> nn.Module:
    """
    Congela todos los parámetros salvo los de la última capa lineal.

    Args:
        model: modelo de clasificación.

    Returns:
        El mismo modelo, con ``requires_grad=False`` en todo el *backbone* y
        ``requires_grad=True`` en la cabeza.
    """

    # BEGIN SOLUTION
    for param in model.parameters():
        param.requires_grad = False
    _, head = get_final_layer(model)
    for param in head.parameters():
        param.requires_grad = True
    return model
    # END SOLUTION


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

    # BEGIN SOLUTION
    for name, param in model.named_parameters():
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            param.requires_grad = True
    return model
    # END SOLUTION


def get_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """
    Devuelve los parámetros con ``requires_grad=True`` (los que verá el optimizador).

    Args:
        model: modelo.

    Returns:
        Lista de parámetros entrenables, en el orden de ``model.parameters()``.
    """

    # BEGIN SOLUTION
    return [p for p in model.parameters() if p.requires_grad]
    # END SOLUTION


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Cuenta el número de escalares de los parámetros de un modelo.

    Args:
        model: modelo.
        trainable_only: si es ``True`` solo cuenta los entrenables.

    Returns:
        Número total de parámetros (``sum(p.numel())``).
    """

    # BEGIN SOLUTION
    params = get_trainable_parameters(model) if trainable_only else model.parameters()
    return sum(p.numel() for p in params)
    # END SOLUTION


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

    # BEGIN SOLUTION
    n = len(layer_groups)
    buckets: list[list[nn.Parameter]] = [[] for _ in range(n)]
    rest: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        for idx, prefix in enumerate(layer_groups):
            if name == prefix or name.startswith(prefix + "."):
                buckets[idx].append(param)
                break
        else:
            rest.append(param)

    groups: list[dict] = []
    if rest:
        groups.append({"params": rest, "lr": base_lr * decay**n})
    for idx, params in enumerate(buckets):
        if params:
            groups.append({"params": params, "lr": base_lr * decay ** (n - 1 - idx)})
    return groups
    # END SOLUTION
