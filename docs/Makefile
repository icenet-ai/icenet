# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS                =
SPHINXBUILD               = python -msphinx
SPHINXMULTIVERSIONBUILD   = python -msphinx_multiversion
SPHINXPROJ                = icenet
SOURCEDIR                 = .
BUILDDIR                  = _build


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help html multiversion Makefile

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Output multiple versions of IceNet docs across history
multiversion:
	@$(SPHINXMULTIVERSIONBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
