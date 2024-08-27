
SHELL := /bin/bash

.SILENT: build clean devenv docs publish test lint
.IGNORE: clean
.ONESHELL:

BLUE:=\033[0;34m
RED:=\033[0;31m
NC:=\033[0m # No Color
BOLD:=$(tput bold)
NORM:=$(tput sgr0)

# the location of this file.
DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# build the package from the source
build: devenv
	. venv/bin/activate;
		python -m build;
		twine check --strict dist/*

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf kafe2.egg-info/
	rm -rf venv/
	rm -rf `find . -type d -name __pycache__`
	cd doc && $(MAKE) clean
	rm -f test-*.yml

devenv:
	if [ ! -d "$(DIR)/venv" ]; then
		echo "Creating venv";
		python -m venv venv/;
	fi
	@if ! venv/bin/python -c "import kafe2" 2>/dev/null; then
		echo "Installing kafe2 in editable mode";
		. venv/bin/activate;
		pip install --upgrade -e .[dev];
	fi

docs:	devenv
	echo "Generating Docs"
	. venv/bin/activate;
		cd doc;
		$(MAKE) html;
		$(MAKE) latex;

publish: build
	echo "uploading build to PyPI"
	. venv/bin/activate;
		twine upload ./dist/*

test: devenv
	echo "Running Pytest and Coverage"
	. venv/bin/activate;
		pytest;
		coverage run

lint: devenv
	. venv/bin/activate;
		echo -e "$(BLUE)${BOLD}ISORT${NC}$(NORM)";
		isort --check --diff ./kafe2;
		RET_Isort=$$?;
		echo -e "$(BLUE)${BOLD}BLACK${NC}$(NORM)";
		black --check --color --diff ./kafe2;
		RET_Black=$$?;
		echo -e "$(BLUE)${BOLD}FLAKE8${NC}$(NORM)";
		flake8 --config .flake8 ./kafe2;
		RET_Flake8=$$?;
		if [ $$RET_Isort -ne 0 ] || [ $$RET_Black -ne 0 ] || [ $$RET_Flake8 -ne 0 ]; then
			echo -e "$(RED)${BOLD}Linting failed${NC}$(NORM)";
			exit 1;
		fi
