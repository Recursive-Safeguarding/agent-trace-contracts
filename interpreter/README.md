# rs_metalang_ref

`rs_metalang_ref` is a runnable reference interpreter for part of the
trace-contracts specification. It executes the worked bounded-response and residual-comparison
fragment. Its modules for indexed evidence, finite-discrete risk, and viability
expose limited utilities or typed distinctions outside that fragment. Its proof
firewall module exposes the no-witness firewall and finite probability
arithmetic, but no concrete shield instance (a shield is a runtime enforcement
component acting on comparison results; none is included here).

## Specification

The normative reference for v0.1.0 is a separate document titled Meta-Language Specification. This non-normative package does not include that document. The S-labels used below and throughout this package are local traceability labels, not section numbers in that document.

## Start with one example

The `conjunction-same-subject` fixture starts with two literal evidence records.
The first record contains proposition `A(m)` with modality `P`. The second
contains proposition `B(m)` with modality `X`. Both use the same model
configuration and index.

| Symbol | Literal value in the test | Meaning in this fixture |
|---|---|---|
| `P` | `Grade.P` | Proof, replay, or checker-validated deductive evidence. |
| `X` | `Grade.X` | Randomised or otherwise identified interventional evidence. |
| `A` | `A` in `A(m)` | The first proposition label. The fixture gives it no domain-specific meaning. |
| `B` | `B` in `B(m)` | The second proposition label. The fixture gives it no domain-specific meaning. |
| `m` | `m` in both propositions | The model-family value and shared argument. |
| shared index | subject `(m, h1, p, tools, scaffold, deploy)`; version `v1`; time `t1`; intervention `i1`; population `pop1` | The fields that identify what both evidence records concern. |

The test calls `and_intro` and expects proposition `A(m) and B(m)`. Its
requirement, `P[A] AND X[B]`, retains the modality of each input. The returned
index must equal the shared input index.

Run only this fixture from the package directory:

```sh
uv run --frozen python -m pytest -p no:cacheprovider -q tests/test_s2_evidence.py::test_s2_conjunction_same_subject
```

## Running the test suite

From this directory, run:

```bash
uv run --frozen python -m pytest -p no:cacheprovider -q
```

Passing this command provides test evidence for the implemented cases mapped
below. Coverage is limited to the listed test surfaces. To list the collected
tests, run
`uv run --frozen python -m pytest -p no:cacheprovider -q --collect-only`.

## Implemented surfaces

The table below names the surfaces that this package exposes. Module docstrings
give the local API details. Labels `S1` through `S6` are local implementation
names; suffixes such as `.4` are traceability labels, not section numbers in
the specification. The source map below associates each specification topic and S-label with its
related tests.

| Surface | Included here | Boundary |
|---|---|---|
| Contract monitor | The single `after TRIGGER when GUARD require RESPONSE within D` clause used by the S1 total three-valued bounded-response semantics fixtures, with linear response allocation and typed terminal conversion | There is no general pattern-matching compiler or general multi-clause monitor |
| Discharge mode | `Linear()` in `SingleClauseMonitor` | Constructing the monitor with `Broadcast(key)` raises `UnsupportedDischargeModeError`; the call never falls through to linear allocation |
| Event input | Domain, tick, and terminal events used by the monitor | `MalformedEvent` is outside this interpreter's event boundary; `SingleClauseMonitor.step(...)` raises `UnsupportedEventTypeError` before it reads or changes state |
| Indexed evidence | Subject unification, conjunction, duplicate-report fusion, and the implemented identifiability checks | Coverage is limited to the listed operations under the fixture-declared dependence model |
| Finite-discrete risk | `ThreeWayReport`, `AmbiguitySet`, and its finite-discrete upper-risk calculation | The package does not encode the general set of joint measures, a policy gate, action selection, or proof-based set elimination |
| Viability | Finite-state fixpoint and classification utilities | The abstraction or model-checking front end is outside this package; the infinite-horizon stochastic result and its `AlmostSure` classification are outside the executable reference fragment, and the finite tests bound a numerical estimate only |
| Residual comparison | `TotalizedLTS`, `monitor_residual`, and bounded word search over unit data with one relative tick per label | `bounded_compare` returns `Distinguished`, `NoWitnessWithinBound`, or `SearchIncomplete`; it never returns `ProvedEquivalent` and does not accept caller-defined data domains or timing schedules. The normative Specification defines `ReplayReceipt`, but this runnable package has no replay receipt producer |
| Result algebra | Five comparison-result constructors and their scope records | Result rendering and linting are outside this executable reference package |
| Proof firewall | The implemented result-type separation checks and probability-bound arithmetic | The package contains no shield-certificate checker, no shield-action selector, and no executable hard-safety guarantee |

The boundaries in this README and the module docstrings describe
implementation coverage only; they do not add to or settle anything in the
specification.

## Source map

| Specification topic and local traceability label | Related tests |
|---|---|
| Bounded-response acceptance tests, worked example, and failure case (S1) | `tests/test_s1_bounded_response.py` |
| Indexed-evidence acceptance tests, worked example, and failure case (S2) | `tests/test_s2_evidence.py` |
| Finite-discrete-risk acceptance tests, worked example, and failure case (S3) | `tests/test_s3_finite_discrete_risk.py` |
| Viability-fixpoint acceptance tests, worked example, and failure case (S4) | `tests/test_s4_viability.py` |
| Contract-relative continuation equivalence: acceptance tests, worked example, and failure case (S5) | `tests/test_s5_residual.py` and `tests/test_s5_acceptance_disabled_sink_retains_observation.py` |
| Firewall clauses, the probabilistic form, and acceptance tests (S6) | `tests/test_s6_firewall.py` |
| Result-algebra constructors and comparison scope, what each result may discharge, and the closed constructor vocabulary | `tests/test_s5_result_scope.py` and `tests/test_s5_invalid_comparison_request.py` |
| Typed events, the totalised transition system, and the residual (S1, S5) | `tests/test_s5_contract_hardening.py` |
| Response-token discipline (S1.4) | `tests/test_linear_explicit_discharge.py` and `tests/test_allocation_properties.py` |
| Typed events and response-token discipline (S1.1, S1.4) | `tests/test_conditional_and_mode_boundaries.py` |
