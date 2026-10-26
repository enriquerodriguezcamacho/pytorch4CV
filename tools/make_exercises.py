"""Genera ``exercises/**.py`` a partir de ``solutions/**.py``.

En las soluciones, el código que debe escribir el alumno va entre marcadores::

    # BEGIN SOLUTION <pista opcional>
    ...código...
    # END SOLUTION

y en el ejercicio se sustituye por::

    # TODO: <pista opcional>
    raise NotImplementedError("TODO")

Uso::

    python tools/make_exercises.py          # regenera todos los ejercicios
    python tools/make_exercises.py --check  # falla si alguno está desactualizado
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN = re.compile(r"^(?P<indent>\s*)# BEGIN SOLUTION\s*(?P<hint>.*)$")
END = re.compile(r"^\s*# END SOLUTION\s*$")


def strip_solution(source: str) -> str:
    out: list[str] = []
    inside = False
    for line in source.splitlines():
        if not inside and (match := BEGIN.match(line)):
            inside = True
            indent, hint = match["indent"], match["hint"].strip()
            out.append(f"{indent}# TODO: {hint or 'implementa esta parte'}")
            out.append(f'{indent}raise NotImplementedError("TODO")')
        elif inside and END.match(line):
            inside = False
        elif not inside:
            out.append(line)
    if inside:
        raise ValueError("Falta un '# END SOLUTION'")
    text = "\n".join(out) + "\n"
    return text.replace("— soluciones.", "— ejercicios.", 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for sol in sorted((ROOT / "solutions").rglob("*.py")):
        rel = sol.relative_to(ROOT / "solutions")
        exe = ROOT / "exercises" / rel
        new = strip_solution(sol.read_text(encoding="utf-8"))
        old = exe.read_text(encoding="utf-8") if exe.exists() else None
        if new != old:
            stale.append(rel)
            if not args.check:
                exe.parent.mkdir(parents=True, exist_ok=True)
                exe.write_text(new, encoding="utf-8", newline="\n")
    for rel in stale:
        print(("DESACTUALIZADO: " if args.check else "generado: ") + str(rel))
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
