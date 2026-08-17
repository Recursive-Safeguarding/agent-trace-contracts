# Worked example

This package is the demonstration that accompanies the trace-contracts
specification. It instantiates one profile, `profile-p-export-approval-k2-v1`,
and runs it through the sibling reference interpreter in `interpreter/`.

The example takes three records of the same short agent history: the full
record, a one-event tail, and an obligation table. It compares the full record
against each shortened one, so the command prints two results as JSON. The
repository's top-level README walks through what those comparisons find and
why. The test suite also checks that two- and three-entry tail windows return
`NoWitnessWithinBound` at the same bounded scope.

Run the example from this directory:

```sh
uv run --frozen python -I -m rs_capsule
```

Run the acceptance tests:

```sh
uv run --frozen python -m pytest -p no:cacheprovider -q
```
