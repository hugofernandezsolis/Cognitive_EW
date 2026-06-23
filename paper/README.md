# Paper — Cognitive Electronic Warfare using Deep RL

Integrated cognitive EW manuscript (single integrated paper, English, `IEEEtran` journal class,
venue-neutral: no target journal is named in the manuscript so it can be submitted to several).

## Files

- `main.tex` — the manuscript. Sources every number from `tables/*.tex` (via `\input`) and
  every included plot from `figs/*.pdf` (via `\includegraphics`); two schematics are inline
  TikZ.
- `refs.bib` — bibliography (re-verify paywalled entries against the source before submission;
  several entries carry placeholder author fields flagged below).
- `IEEEtran.cls` — the IEEE journal class, kept beside `main.tex` so it compiles anywhere
  (Overleaf already ships this class).
- `tables/` — generated LaTeX table bodies. **Do not edit by hand.**
- `figs/` — generated grayscale-safe figures. **Do not edit by hand.** `anchor_results.pdf`
  is generated as an optional summary figure but is not currently included in `main.tex`.

## Build (Overleaf or local)

Overleaf: upload the `paper/` folder, set `main.tex` as the main document, compiler `pdfLaTeX`.

Local:

```bash
cd paper
latexmk -pdf main.tex      # or: pdflatex; bibtex; pdflatex; pdflatex
```

The current dev environment has `latexmk`/`pdflatex` available; the manuscript has been checked
with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.

## Regenerate tables and figures from results

Both are deterministic and read only `results/anchors_full_ms_accepted/` (the five-seed
campaign); tables/figures report mean $\pm$ std over seeds:

```bash
# from the repository root
uv run python                  scripts/paper/extract_results_tables.py
uv run --with matplotlib python scripts/paper/build_figures.py
```

## Before submission — remaining manual steps

- **References:** entries have been checked against DOI/Crossref, publisher pages, TechRxiv or
  Google Patents metadata where available. Before final submission, do one last publisher-side
  check for paywalled/preprint records whose metadata may change (`wdcgan2026`,
  `jammervsradar2026`).
- **Author block:** confirm the exact `Projener.ai` capitalization and affiliation address.
- **Numbers are honest by construction:** the ELINT anchor passes on accuracy (on all five
  seeds) but not on the $<1$\,ms $p99$ latency target; this is reported as a limitation/future
  work and must not be "rounded up." See `results/anchors_full_ms_accepted/README.md`.
- The title retains the long NTTR phrasing; the body consistently frames the range as
  *NTTR-inspired and synthetic*.
- **Optional figure decision:** `anchor_results.pdf` can be added if the paper needs a quick visual
  summary of the anchor gates. It duplicates part of Table I, so include it only if page budget and
  narrative flow benefit from a visual scoreboard.
