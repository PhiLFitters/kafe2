# Makefile to build Sphinx documentation in `src' subdirectory
# Run `make' to compile the documentation in HTML and LaTeX (PDF) format


# Variables

# subdirectory with Sphinx Makefile
SPHINXSRCDIR = src


# Make Targets

.PHONY: all html latex

all: html latex

html:
	$(MAKE) -C $(SPHINXSRCDIR) html && rm -rf html && cp -R "$(SPHINXSRCDIR)/_build/html" .

latex:
	$(MAKE) -C $(SPHINXSRCDIR) latex && $(MAKE) -C $(SPHINXSRCDIR) latexpdf && rm -rf latex && cp -R "$(SPHINXSRCDIR)/_build/latex" .

clean:
	$(MAKE) -C $(SPHINXSRCDIR) clean && rm -rf html latex

forceclean:
	rm -rf src/_build

