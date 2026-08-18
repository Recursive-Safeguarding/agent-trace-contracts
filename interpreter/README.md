# rs_metalang_ref

`rs_metalang_ref` executes rules about what an agent owes. Give it a history
("exported at tick 0, approved at tick 1") and a rule ("every export needs an
approval within two ticks"), and it reports whether the rule is satisfied,
violated, or not yet decided, and whether the run is still going.

It does that for one shape of rule: a bounded response, meaning an obligation
that must be discharged within a fixed number of ticks. It also compares two
histories to see whether a checker can tell them apart, which is the
residual comparison the worked example uses.

Its other modules cover narrower ground. Indexed evidence, finite-discrete
risk, and viability expose limited utilities or typed distinctions outside that
fragment. The proof firewall module exposes the no-witness firewall and finite
probability arithmetic, but no concrete shield instance (a shield is a runtime
enforcement component acting on comparison results; none is included here).

## Specification

The normative reference for v0.1.0 is a separate document, the Meta-Language
Specification, which this package does not include. The `S1` to `S6` labels
used throughout this package, and their dotted forms such as `S1.4` and
`S5.7`, point at topics in the specification without being its section
numbers.

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

The suite exercises the cases mapped in the source map below, alongside further
tests the map does not list. To list the collected tests, run
`uv run --frozen python -m pytest -p no:cacheprovider -q --collect-only`.

## Implemented surfaces

The table below names the surfaces that this package exposes; module docstrings
give the local API details. The "Boundary" column states the scope each surface
is valid within, which is a fact about this implementation rather than about the
specification. The source map at the end of this README associates the
S-labels it lists with their related tests.

| Surface | Included here | Boundary |
|---|---|---|
| Contract monitor | The single `after TRIGGER when GUARD require RESPONSE within D` clause used by the S1 total three-valued bounded-response semantics fixtures, with linear response allocation and typed terminal conversion | There is no general pattern-matching compiler or general multi-clause monitor |
| Discharge mode | `Linear()` in `SingleClauseMonitor` | Constructing the monitor with `Broadcast(key)` raises `UnsupportedDischargeModeError` rather than falling back to linear allocation |
| Event input | Domain, tick, and terminal events used by the monitor | `MalformedEvent` is outside this interpreter's event boundary; `SingleClauseMonitor.step(...)` raises `UnsupportedEventTypeError` before it reads or changes state |
| Indexed evidence | Subject unification, conjunction, duplicate-report fusion, and the implemented identifiability checks | Coverage is limited to the listed operations; fusion runs only under a declared `Duplicate` dependence |
| Finite-discrete risk | `ThreeWayReport`, `AmbiguitySet`, and its finite-discrete upper-risk calculation | It takes the ambiguity set as given rather than deriving or narrowing it, and computes a risk bound rather than gating on it |
| Viability | Finite-state fixpoint and classification utilities | The tests bound a numerical estimate rather than proving the infinite-horizon stochastic result |
| Residual comparison | `TotalizedLTS`, `monitor_residual`, and bounded word search over unit data with one relative tick per label | `bounded_compare` returns `Distinguished`, `NoWitnessWithinBound`, or `SearchIncomplete`, and never `ProvedEquivalent`: a search that finds no witness within its bound has not proved equivalence |
| Result algebra | Five comparison-result constructors and their scope records | Result rendering and linting are outside this package |
| Proof firewall | The implemented result-type separation checks and probability-bound arithmetic | The package contains no shield-certificate checker, no shield-action selector, and no executable hard-safety guarantee |

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
