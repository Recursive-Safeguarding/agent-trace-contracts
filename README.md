<p align="center"><img src="docs/assets/logo.svg" width="96" alt="agent-trace-contracts logo"></p>

# Trace-contracts reference implementation

<p align="center">
  <img alt="Python 3.11 to 3.13" src="https://img.shields.io/badge/python-3.11%20to%203.13-3776AB?logo=python&logoColor=white">
  <img alt="uv" src="https://img.shields.io/badge/packaging-uv-6340AC">
  <img alt="pytest" src="https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white">
  <img alt="MIT licence" src="https://img.shields.io/badge/licence-MIT-165354">
  <img alt="version 0.1.0" src="https://img.shields.io/badge/version-0.1.0-165354">
</p>

An agent exports something at tick 0 and receives an approval at tick 1. It
exports again at tick 2, then the clock advances to tick 3 with no second
approval. A rule in force says every export must be approved within two ticks,
so one approval is still owed.

Now shorten that history, as a long-running agent must. Keep only the last
event and the owed approval disappears; keep a table of what is still owed and
it survives. **This software searches for evidence that a shortened record
answers a question differently from the full one**, by running both through
the futures the profile admits, out to a declared bound, and comparing what a
checker reads.

The worked example below runs exactly that comparison. `interpreter/` is the
engine; `abstraction-capsule/` is the example.

## Specification

The normative specification for v0.1.0 is a separate document titled
Meta-Language Specification. It accompanies the submission this
repository supports and is not published separately, so this repository does
not include it.

The `interpreter/` package executes the worked bounded-response and
residual-comparison fragment. It also exposes limited utilities for indexed
evidence, finite-discrete risk, viability, the no-witness firewall, and finite
probability arithmetic. The `abstraction-capsule/` package runs one worked
comparison through that interpreter. The repository contains no concrete
shield instance (a shield is a runtime enforcement component that would act
on comparison results; none is included here).

## The two packages

| Directory | Package | What it is |
|---|---|---|
| `interpreter/` | `rs_metalang_ref` | The reference interpreter. It executes the worked bounded-response and residual-comparison fragment. Its README lists the other surfaces it exposes. |
| `abstraction-capsule/` | `rs-capsule-acceptance` | One worked example. It uses the interpreter to compare three records of the same short history under a single profile, the full record against each of the other two, and prints what it found. |

Start with `abstraction-capsule/`. It runs the worked comparison end to end
before you inspect the interpreter implementation.

**The two directories must stay siblings.** The capsule's lockfile records the
interpreter at the relative path `../interpreter` and installs it from source.
Move either directory and the lockfile no longer resolves.

## Installing

You need Python 3.11 to 3.13, and [uv](https://docs.astral.sh/uv/), the Python
package manager these packages are locked against. On macOS or Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

There is nothing else to install. Each `uv run` command below reads the
package's lockfile, builds an isolated environment on first use, and runs inside
it. You do not need to create a virtual environment or install anything by hand.

## Running the worked example

The example runs the history from the top of this README, under a profile
requiring an approval within two ticks of each export. The source record
therefore contains one open approval obligation. The one-event tail drops it,
while the obligation-table record keeps it.

Suppose the run then ends. The interpreter appends `Complete`, the terminal
event marking a finished run, and reads each record again: the source reports
`Violated/Complete`, because the owed approval never arrived, while the tail
reports `Satisfied/Complete`, because it no longer holds the obligation to
breach. The two records disagree, so the interpreter returns `Distinguished`.
For the obligation-table record, the interpreter checks all seven continuations
of at most two events and finds no such disagreement, so it returns
`NoWitnessWithinBound`: no witness was found within the scope searched, which
is a weaker statement than equivalence. The comparison observes only
`(Summary, Mode)`, under this one profile.

The test suite also checks that two- and three-entry tail windows return
`NoWitnessWithinBound` at the same bounded scope.

```sh
(
  cd abstraction-capsule
  uv run --frozen python -I -m rs_capsule
)
```

The command prints a JSON array holding one record per comparison.

## Running the test suites

Each package carries its own suite, run from its own directory:

```sh
(
  cd interpreter
  uv run --frozen python -m pytest -p no:cacheprovider -q
)
(
  cd abstraction-capsule
  uv run --frozen python -m pytest -p no:cacheprovider -q
)
```

Each command exits with status 0 when its suite passes. If a command fails,
its error output identifies the failing test.

The suites exercise implemented cases. Passing them does not by itself establish
full conformance to the specification. The source map in
`interpreter/README.md` associates test modules with their
specification topics, most of them carrying an S-label, a local traceability
marker rather than a specification section number.

## Licence and provenance

This repository is maintained by Recursive Safeguarding Ltd
([recursive-safeguarding.org](https://recursive-safeguarding.org)).
The code is available under the MIT License. See `LICENSE`.
