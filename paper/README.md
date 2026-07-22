# Paper

`paper.tex` is the preprint draft for the research study in [`../research/`](../research/).

The compiled [`paper.pdf`](paper.pdf) (8 pages) is committed for convenience.

## Build

Recommended — [Tectonic](https://tectonic-typesetting.github.io/) (self-contained,
fetches packages on demand, resolves refs in one pass):

```bash
cd paper
tectonic paper.tex        # brew install tectonic
```

Or a stock TeX Live:

```bash
cd paper
pdflatex paper.tex && pdflatex paper.tex   # twice, to resolve references
```

It uses only standard packages (`geometry`, `graphicx`, `booktabs`, `amsmath`,
`amssymb`, `siunitx`, `hyperref`, `caption`, `xcolor`) and pulls figures from
`../docs/img/`.

## Regenerating the results the paper cites

```bash
python -m research.run_all            # full study (hours)
python -m research.run_all --smoke    # fast sanity pass
```

Numbers quoted in the text live in `../research/results/*.json`; update the
manuscript from those after a full run.
