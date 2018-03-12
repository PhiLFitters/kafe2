# Makefile to build Sphinx documentation in `build' subdirectory
# Run `make' to compile the documentation in HTML and LaTeX (PDF) format


# Variables

# subdirectory with Sphinx Makefile
SPHINXSRCDIR = build


# Make Targets

.PHONY: all html latex

all: html latex

html:
	$(MAKE) -C $(SPHINXSRCDIR) html && rm -rf html && cp -R "$(SPHINXSRCDIR)/_build/html" .

latex:
	$(MAKE) -C $(SPHINXSRCDIR) latex && $(MAKE) -C $(SPHINXSRCDIR) latexpdf && rm -rf latex && cp -R "$(SPHINXSRCDIR)/_build/latex" .

clean:
	rm -rf build/_build

