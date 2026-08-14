PYTHON ?= python3
LATEX ?= pdflatex
BIBTEX ?= bibtex
ICRA_CLASS_DIR ?= .
REPO_ROOT := $(abspath ../..)
TEXFLAGS := -interaction=nonstopmode -halt-on-error -output-directory=build

.PHONY: all figures icra clean

all: figures icra

figures:
	cd $(REPO_ROOT) && $(PYTHON) paper/icra2027_cpmd/scripts/plot_existing_results.py
	cd $(REPO_ROOT) && $(PYTHON) paper/icra2027_cpmd/scripts/compose_roll_teaser.py

define build_tex
	@mkdir -p build
	TEXINPUTS=$(ICRA_CLASS_DIR):.: $(LATEX) $(TEXFLAGS) $(1).tex
	BIBINPUTS=.: BSTINPUTS=.: $(BIBTEX) build/$(1)
	TEXINPUTS=$(ICRA_CLASS_DIR):.: $(LATEX) $(TEXFLAGS) $(1).tex
	TEXINPUTS=$(ICRA_CLASS_DIR):.: $(LATEX) $(TEXFLAGS) $(1).tex
endef

icra:
	$(call build_tex,main_icra)

clean:
	rm -f build/*.aux build/*.bbl build/*.blg build/*.log build/*.out build/*.pdf
