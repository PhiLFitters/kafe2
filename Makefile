
SHELL := /bin/bash

.SILENT: clean lint
.IGNORE: clean

BLUE:=\033[0;34m
NC:=\033[0m # No Color
BOLD:=$(tput bold)
NORM:=$(tput sgr0)

# build the package from the source
build:
	python -m build
	twine check --strict dist/*

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf kafe2.egg-info/
	rm -rf venv/
	rm -rf `find . -type d -name __pycache__`
	cd doc && $(MAKE) clean
	rm -f test-*.yml

# create a development environment
devenv:	build
	python -m venv venv/
	. venv/bin/activate; pip install -e .[dev]

docs:	build devenv
	. venv/bin/activate; cd doc && $(MAKE) html

publish: build docs
	twine upload ./dist/*

test: build
	pytest
	coverage run

lint:
# $make devenv must be called before this and the venv has to be activated.
# Otherwise the packages isort, black and flake8 might be missing.
	echo -e "$(BLUE)${BOLD}ISORT${NC}$(NORM)"
	isort --check --diff ./kafe2

	echo -e "$(BLUE)${BOLD}BLACK${NC}$(NORM)"
	black --check --diff ./kafe2

	echo -e "$(BLUE)${BOLD}FLAKE8${NC}$(NORM)"
	flake8 --config .flake8 ./kafe2
