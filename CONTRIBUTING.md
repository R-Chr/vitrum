# Contributing to vitrum

Contributions are welcome — bug reports, reproducible test cases, documentation fixes and
new analysis routines alike. This page covers how to get set up and what a mergeable change
looks like.

## Reporting a bug

Open an issue at https://github.com/R-Chr/vitrum/issues. For anything that produces a wrong
number rather than an exception, please include the structure or trajectory (or a small
script that generates one), the exact call you made, and the number you expected — a
scientific bug report is only actionable if the result can be reproduced.

Before filing, check [known issues](docs/vitrum/known_issues.md); several limitations are
documented there deliberately, including the non-periodic persistent-homology filtration,
the orthorhombic-only routines, and the unsupported `vitrum.batch_active` module.

## Asking a question

Use the issue tracker for questions too. If the answer turns out to be missing from the
docs, that is a documentation bug and we would rather fix it than answer it twice.

## Development setup

```bash
git clone https://github.com/R-Chr/vitrum.git
cd vitrum
pip install -e ".[test]"
```

`vitrum` supports Python 3.10 and newer. The optional extras (`workflows`,
`volume_estimation`, `persistent_homology`, `visualization`) are not needed for the test
suite — see [the install docs](docs/vitrum/install.md) if you are working on code behind one
of them, since `persistent_homology` additionally needs CGAL and a manual `diode` install.

## Running the tests

```bash
pytest
```

The suite takes well under a minute. Every change should leave it green; run it before
opening a pull request rather than relying on CI to tell you.

## Style and linting

Formatting and linting are handled by [ruff](https://docs.astral.sh/ruff/), configured in
`pyproject.toml`. CI enforces both, so run them locally:

```bash
pip install ruff
ruff check src tests
ruff format src tests
```

Or install the hooks and forget about it:

```bash
pip install pre-commit
pre-commit install
```

Public functions and classes carry Google-style docstrings with an `Args:` and `Returns:`
section, and type annotations on the signature. New public API should match.

## Type annotations

`vitrum` ships a `py.typed` marker, so its annotations are a promise to downstream users'
type checkers. CI enforces that promise with `mypy` under `disallow_untyped_defs` and
`disallow_incomplete_defs`, configured in `pyproject.toml`:

```bash
pip install mypy
mypy
```

## Pull requests

- Branch off `main` and keep the change focused on one thing.
- Add a test that fails without your change. For numerical work this matters most: a fix to
  a formula needs a case whose correct answer is known independently, not one whose expected
  value was read off the new implementation.
- Update the docs under `docs/` when you change behaviour, and add an entry to
  `CHANGELOG.md` under an `## [Unreleased]` heading.
- If your change alters numbers that previous versions returned, say so explicitly in the
  changelog entry — users need to know when to recompute.
- Formatting-only changes are best kept in their own commit, separate from logic.

## Releases

`vitrum` follows [semantic versioning](https://semver.org/) as of 1.0. Breaking API changes
wait for a major bump. `__version__` in `src/vitrum/__init__.py` and `version:` in
`CITATION.cff` must agree — a test enforces it.

## Code of conduct

Participation in this project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
