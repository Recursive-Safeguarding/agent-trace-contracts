# Trace-contracts reference code and test evidence

This repository contains Recursive Safeguarding Ltd's non-normative reference
code and test evidence for its trace-contracts meta-language specification.

## Specification

The normative specification for v0.1.0 is a separate document titled
Meta-Language Specification. This non-normative repository does not include
that document.

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
| `abstraction-capsule/` | `rs-capsule-acceptance` | One worked example. It uses the interpreter to compare two records of the same short history under a single profile, and prints what it found. |

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

An agent exports at tick 0 and receives approval at tick 1. It exports again at
tick 2, then time advances to tick 3 without another approval. This profile
requires an approval within two ticks of each export. The source record
therefore contains one open approval obligation. The one-event tail drops it,
while the obligation-table record keeps it.

If `Complete` is appended, the source becomes `Violated/Complete` and the tail
becomes `Satisfied/Complete`,
so the interpreter returns `Distinguished`. The test suite also checks that two- and
three-entry tail windows return `NoWitnessWithinBound` at the same bounded
scope. For the obligation-table record,
the interpreter checks all seven continuations of at most two events and
returns `NoWitnessWithinBound`. The comparison observes only `(Summary, Mode)`;
it does not establish full-observation equivalence or agreement for longer
continuations or another profile.

```sh
(
  cd abstraction-capsule
  uv run --frozen python -I -m rs_capsule
)
```

The command prints one JSON record for each comparison.

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

Passing both suites provides the repository's test evidence for the implemented
cases. It does not by itself establish full conformance to the specification. The
source map in `interpreter/README.md` associates each main test module with its
specification topic and an S-label; the labels are local traceability markers,
not specification section numbers.

## Licence and provenance

This repository is maintained by Recursive Safeguarding Ltd
([recursive-safeguarding.org](https://recursive-safeguarding.org)).
The code is available under the MIT License. See `LICENSE`.
