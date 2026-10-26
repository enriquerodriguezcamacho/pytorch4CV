# pytorch4CV — atajos habituales (mismo espíritu que el Makefile de los exámenes)
.PHONY: install exercises check test test-solutions lint format clean
.DEFAULT_GOAL := test

PYTHON ?= python
TARGET ?= exercises

install:            ## Instala dependencias con uv (torch CPU)
	uv sync --extra dev

exercises:          ## Regenera exercises/ a partir de solutions/
	$(PYTHON) tools/make_exercises.py

check:              ## Comprueba que exercises/ está sincronizado
	$(PYTHON) tools/make_exercises.py --check

test:               ## Ejecuta los tests contra tus ejercicios
	$(PYTHON) -m pytest --target $(TARGET)

test-solutions:     ## Ejecuta los tests contra las soluciones
	$(PYTHON) -m pytest --target solutions

lint:               ## Linters (ruff + mypy sobre las soluciones)
	ruff check .
	ruff format --check .
	mypy solutions --ignore-missing-imports

format:             ## Formatea el código
	ruff check --fix .
	ruff format .

clean:              ## Borra cachés
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
