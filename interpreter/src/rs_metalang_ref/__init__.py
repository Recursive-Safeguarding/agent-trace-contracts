"""Runnable companion interpreter for the trace-contracts meta-language specification.

This package executes the worked bounded-response and residual-comparison
reference fragment (S1 and S5). Its modules for indexed evidence,
finite-discrete risk, and viability (S2-S4) expose partial utilities or typed
distinctions outside that fragment. The proof firewall module (S6) exposes the
no-witness firewall and finite probability arithmetic, but no concrete shield
instance. The specification document remains normative.

The `S1.4`-style labels below and throughout the modules are local
implementation labels, not specification section identifiers. README.md maps
each construction to its PDF section.

Module map:
    kleene      -- S1.2 strong-Kleene three-valued truth domain
    events      -- S1.1 normalized monitor input events
    contracts   -- S1.1/S1.5 contract grammar and per-clause-form evaluators
    obligations -- S1.4 response-obligation lifecycle
    verdict     -- S1.3/S1.8/S1.9 public verdict object and permission judgement
    monitor     -- S1.6/S1.7 total transition order and terminal conversion
    evidence    -- S2 indexed evidence and claim system
    robust      -- S3 finite-discrete robust-risk calculation
    viability   -- S4 viability fixpoint
    residual    -- S5 total residual definition and bounded comparison
    firewall    -- S6 proof firewall and probability-bound arithmetic
"""
