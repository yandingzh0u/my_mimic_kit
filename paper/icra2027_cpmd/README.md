# Context-Preserving Motion Differentials

This directory contains one anonymous, eight-page ICRA 2027 paper.  The eight
pages include figures, tables, conclusion, and references.  There is no
separate extended manuscript or appendix in this directory.

## Build

```bash
make figures
make icra
```

Download the official ICRA/PaperCept LaTeX template and point
`ICRA_CLASS_DIR` at the directory containing `ieeeconf.cls`.  Then run, for
example:

```bash
make icra ICRA_CLASS_DIR=/path/to/icra-template
```

`LATEX`, `BIBTEX`, and `PYTHON` may be overridden for a local toolchain.  The
figure target regenerates plots from the local experiment artifacts and
recomposes already captured simulator frames; it never launches training.
No figure is copied from another paper.

## Reproducibility status

The currently filled Roll values are the frozen seed-0 results.  The primary
L1 and CPMD checkpoints were each evaluated for 256 episodes of 10 s.  The
paper explicitly separates these completed pilot results from the multi-seed,
matched-width, and multi-motion experiments still required for submission.

The figure scripts read existing logs and checkpoints only; they never launch
training.  Every derivation, citation, and experimental claim must be checked
by the authors before submission.  Any required author or tool-use disclosure
should be completed in the submission system after de-anonymization.
