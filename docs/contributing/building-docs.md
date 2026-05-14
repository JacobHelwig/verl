# Building the Docs

A minimal workflow for previewing the verl docs locally with [uv](https://docs.astral.sh/uv/).

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed.

## One-time setup

Create a virtual env in `docs/` and install the doc dependencies plus
[`sphinx-autobuild`](https://github.com/sphinx-doc/sphinx-autobuild) (used for
the live-preview step below):

```bash
cd docs
uv venv
uv pip install -r requirements-docs.txt sphinx-autobuild
```

This creates `docs/.venv/`. The repo's `.gitignore` excludes it, so nothing new
needs to be committed.

## One-shot build

```bash
cd docs
.venv/bin/sphinx-build -b html . _build/html
open _build/html/index.html      # macOS; use xdg-open on Linux
```

To start clean: `rm -rf _build`.

## Live preview (recommended for iteration)

```bash
cd docs
.venv/bin/sphinx-autobuild . _build/html
```

Open <http://127.0.0.1:8000>. Saves to any `.md`, `.rst`, or `conf.py` trigger
an incremental rebuild and the browser auto-reloads.

To bind a different port: `--port 8765`. To open the browser automatically:
`--open-browser`.
