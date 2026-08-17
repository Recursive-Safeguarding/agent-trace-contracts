# Worked example

## Specification

The normative reference for v0.1.0 is a separate document titled Meta-Language Specification. This non-normative package does not include that document.

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

Run the example from this directory:

```sh
uv run --frozen python -I -m rs_capsule
```

This directory is the non-normative demonstration package for the specification's worked comparison. The specification defines the relevant language fragment; this program instantiates one profile, `profile-p-export-approval-k2-v1`, using the sibling reference interpreter.

Run the acceptance tests:

```sh
uv run --frozen python -m pytest -p no:cacheprovider -q
```
