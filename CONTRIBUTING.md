# Guía de contribución y de estilo

¡Gracias por querer mejorar **pytorch4CV**! Este documento explica cómo está
organizado cada tema y qué convenciones seguimos, para que todos los ejercicios
se parezcan a los de un examen real de *Computer Vision II*.

## 1. Flujo de trabajo

1. Haz un *fork* y crea una rama: `git checkout -b tema/nombre-corto`.
2. Escribe **solo** en `solutions/` (y en los `README.md` de `exercises/`).
3. Regenera los enunciados: `python tools/make_exercises.py`.
4. Comprueba que todo pasa:
   ```bash
   pytest --target solutions        # las soluciones pasan todos los tests
   pytest tests/<ruta>/test_x.py    # los ejercicios fallan con NotImplementedError
   python tools/make_exercises.py --check
   ruff check . && ruff format --check .
   ```
5. Abre un *Pull Request* describiendo qué tema añades o corriges.

## 2. Estructura de un tema

```
exercises/<bloque>/<NN_tema>/
├── README.md            # teoría + casos de uso + tabla de ejercicios (SOLO aquí)
└── <modulo>.py          # GENERADO a partir de solutions/ (no editar a mano)
solutions/<bloque>/<NN_tema>/
└── <modulo>.py          # código completo con marcadores BEGIN/END SOLUTION
tests/<bloque>/test_<id>_<tema>.py
```

* `<bloque>` es `fundamentals` o `cv`.
* Prefijo de los tests: `f01`, `f02`… para fundamentos y `cv01`, `cv02`… para CV
  (los nombres de fichero de test deben ser únicos en todo el repositorio).
* Un tema de CV puede tener varios ficheros (`model.py`, `loss.py`, `train.py`,
  `utils.py`, `config.py`), igual que en los exámenes. Entre ficheros del mismo
  tema se importa **con imports relativos**: `from .utils import gram_matrix`.

## 3. Marcadores de solución

El script `tools/make_exercises.py` sustituye todo lo que hay entre marcadores por
un `TODO`:

```python
def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    ...docstring completo (se conserva en el ejercicio)...
    """

    # BEGIN SOLUTION
    b, c, h, w = x.shape
    feats = x.view(b, c, h * w)
    return feats @ feats.transpose(1, 2) / (c * h * w)
    # END SOLUTION
```

→ en `exercises/` queda:

```python
    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
```

* Se puede añadir una pista: `# BEGIN SOLUTION usa torch.einsum` → `# TODO: usa torch.einsum`.
* En `__init__`, deja fuera de los marcadores lo que el enunciado "da hecho"
  (por ejemplo `super().__init__()` o la lista de tamaños de canal) y marca solo
  lo que debe escribir el alumno.
* El código **que no aporta a la asignatura** (carga de imágenes, *plots*, guardado)
  se deja sin marcadores, como hacen los exámenes ("some functions are already given").
* Nunca uses sintaxis inválida en el ejercicio (nada de `self.x = # TODO`).

## 4. Estilo del código (igual que en los exámenes)

* Docstring de módulo en líneas propias: `"""\nF05 · Funciones de pérdida — soluciones.\n"""`
  (la primera línea de texto debe terminar en `— soluciones.`; el generador la
  cambia a `— ejercicios.`).
* Docstrings estilo **Google** (`Args:` / `Returns:` / `Raises:`) **en español**,
  indicando siempre las dimensiones de cada tensor:
  `x: imagen de entrada. Dimensions: [batch, channels, height, width].`
* Tipado completo con sintaxis moderna (`int | None`, `tuple[torch.Tensor, torch.Tensor]`,
  `Literal["linf", "l2"]`), `-> None` en `__init__`.
* Configuración de los temas grandes en `config.py` con `@dataclass`.
* Las funciones deben funcionar en **CPU** y en lotes pequeños.
* Líneas ≤ 88 caracteres, imports ordenados (formato `ruff format` / Black).

## 5. Estilo de los tests

Imitamos los tests de los exámenes:

* Docstring de módulo y de cada test; funciones tipadas `-> None`.
* `m = load("cv/02_object_detection/iou.py")` (de `tests/helpers.py`) en vez de
  imports directos: así se puede elegir `--target exercises|solutions`.
* Qué comprobar:
  * forma y `dtype` de la salida;
  * rango de valores (p.ej. `[0, 1]` tras una sigmoide);
  * **valores exactos calculados a mano** con tensores diminutos (`torch.isclose`);
  * comparación con la implementación oficial de PyTorch cuando exista
    (`F.cross_entropy`, `torchvision.ops.nms`...);
  * que una pérdida escala linealmente con su peso (`alpha / 2` → pérdida / 2);
  * que los parámetros cambian tras un paso de entrenamiento y que
    `model.training` es el correcto en *train*/*eval*;
  * `requires_grad`, flujo de gradientes, determinismo con `torch.manual_seed`;
  * errores con `pytest.raises(ValueError)`;
  * `@pytest.mark.parametrize` para barrer tamaños.
* Mensajes en los `assert`: `assert out.shape == (4, 3), "Forma incorrecta."`.
* **Sin descargas** (nada de pesos preentrenados ni datasets) y cada fichero de
  test en < 10 s en CPU. Usa `tests/helpers.DummyCNN` o modelos minúsculos.

## 6. README de cada tema

Plantilla (ver `exercises/fundamentals/01_tensors/README.md`):

1. Título `# CV05 · Explicabilidad` + objetivo + ficheros + comando de test.
2. **Teoría**: conceptos, fórmulas (LaTeX de GitHub `$...$`), diagramas ASCII.
3. **¿Cuándo uso cada cosa?**: tabla de casos de uso reales.
4. **Errores típicos de examen**.
5. **Ejercicios**: tabla `# | Función/clase | Puntos o dificultad (★)`.
6. Pistas en `<details>` y **Referencias** (docs oficiales, papers, repos).

## 7. Código de conducta

Sé amable y constructivo. Las críticas van al código, nunca a las personas.
