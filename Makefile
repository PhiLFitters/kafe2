
SHELL := /bin/bash

.SILENT: clean
.IGNORE: clean

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
