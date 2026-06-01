# Zenodo upload — step-by-step

> _One-time setup to get a DOI for the `PAPER_DRAFT.md` v0.2 archive.
> Zenodo accepts the markdown directly, but reviewers/citers prefer
> PDF. We do both._

## 1. Render the draft to PDF (one command)

```bash
# pandoc → PDF via xelatex (already installed on macOS via MacTeX, or
# `brew install --cask mactex-no-gui` if you don't have it yet)
cd ~/postnet-cf
pandoc docs/PAPER_DRAFT.md \
  -o docs/postnet-cf-v0.2.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V mainfont="Times New Roman" \
  -V monofont="Menlo" \
  -V fontsize=11pt
```

If pandoc isn't installed: `brew install pandoc`. If MacTeX isn't installed
and you don't want to: use `--pdf-engine=weasyprint` (`brew install
weasyprint`) instead — simpler stack, almost as nice.

## 2. Upload to Zenodo

1. Go to <https://zenodo.org> → **Log in** (Google / ORCID / GitHub OAuth all work).
2. Click **+ New Upload**.
3. **Files** — drop in:
   - `docs/postnet-cf-v0.2.pdf` (the rendered paper)
   - `docs/PAPER_DRAFT.md` (the markdown source)
   - Optionally: a zip of `docs/PHASE_*.md` for the per-phase appendix
4. **Metadata** — fill these fields exactly:
   - **Resource type:** Publication → Working paper
   - **Title:** `Postnet-CF: Federated LLM gate-controller training across browser tabs with verified byzantine defense`
   - **Creators:** Ahmet Barış Günaydın (add your ORCID if you have one)
   - **Description:** paste the Abstract from `PAPER_DRAFT.md`
   - **License:** MIT
   - **Keywords:** `federated learning`, `SPSA`, `DeComFL`, `NTK-Mirror`,
     `Cloudflare Workers`, `byzantine fault tolerance`, `browser-tab swarm`,
     `large language models`, `Qwen`
   - **Related identifiers:**
     - `Is supplement to: https://github.com/abgnydn/postnet-cf`
     - `Is supplement to: https://postnet-cf.abgunaydin94.workers.dev`
5. **Communities** (optional): search "Open Research Software" or
   "FAIR4RS" and request inclusion.
6. **Publish.** Confirm. Zenodo issues a DOI immediately, format
   `10.5281/zenodo.XXXXXXX`.

## 3. Add the DOI back to the repo

Once you have the DOI:

```bash
# in CITATION.cff, add this line near the bottom:
#   doi: 10.5281/zenodo.XXXXXXX
# then commit:
git commit -am "add zenodo DOI to citation"
git push
```

Zenodo will auto-display a "cite as" snippet you can paste into
the README and the paper's footer.

## 4. Future versions

Zenodo treats versions as first-class. Next time you update the paper:
- Go to your Zenodo deposit → **New version** → re-upload the new PDF
- The DOI changes (each version has its own) but the **Concept DOI**
  stays constant — that's the one to cite for "the project," any
  version. README and CITATION.cff should reference the concept DOI.

Done. Should take ~10 minutes end-to-end the first time.
