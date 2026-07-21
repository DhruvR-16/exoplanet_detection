# Paper

`paper.tex` is the preprint draft for the research study in [`../research/`](../research/).

## Build

```bash
cd paper
pdflatex paper.tex && pdflatex paper.tex   # twice, to resolve references
```

It uses only stock TeX Live packages (`geometry`, `graphicx`, `booktabs`,
`amsmath`, `siunitx`, `hyperref`, `caption`) and pulls figures from
`../docs/img/`.

## Regenerating the results the paper cites

```bash
python -m research.run_all            # full study (hours)
python -m research.run_all --smoke    # fast sanity pass
```

Numbers quoted in the text live in `../research/results/*.json`; update the
manuscript from those after a full run.
