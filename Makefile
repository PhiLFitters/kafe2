
SHELL := /bin/bash

.SILENT: clean
.IGNORE: clean

build:	clean
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf kafe2.egg-info/
	rm -rf venv/
	rm -rf `find . -type d -name __pycache__`
	#cd docs && $(MAKE) clean

upload: build
	twine upload dist/*

venv:	build #docs
	python -m venv venv/
	. venv/bin/activate; pip install dist/kafe2*.tar.gz
	. venv/bin/activate; pip install ptpython

#docs:	build
	#cd docs && $(MAKE) html
