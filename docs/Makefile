# Minimal Sphinx makefile (docs/ lives at the project root).

SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= uv run sphinx-build
SPHINXAUTOBUILD ?= uv run sphinx-autobuild
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: html livehtml clean help

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

livehtml:
	@$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) --watch ../src

clean:
	@rm -rf "$(BUILDDIR)"
