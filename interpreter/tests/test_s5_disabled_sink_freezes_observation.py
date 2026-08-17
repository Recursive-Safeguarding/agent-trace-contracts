"""The disabled sink must carry the contract observation.

The totalised transition system and residual definitions require a disabled residual to keep
the frozen contract observation. A Satisfied prefix and a Violated prefix must
therefore stay different after a forbidden attempt, while prefixes the contract
does not distinguish still compare equal.
"""

from rs_metalang_ref.residual import TotalizedLTS


def _lts(observations, enabled_set=frozenset({"ok"})):
    """A transition system whose states differ ONLY in what the contract observes.

    Every state enables the same labels and every transition is a self-loop, so the only
    thing that can distinguish two states is the observation. That isolates the property
    under test: nothing else is available to carry the distinction.
    """
    return TotalizedLTS(
        transition=lambda state, label: state,
        enabled=lambda state: enabled_set,
        observe=lambda state: observations[state],
    )


def test_prefixes_distinguished_before_disablement_stay_distinguished():
    """The core requirement. A Satisfied prefix and a Violated one must not become equal
    merely because the continuation attempted something the contract forbids."""
    lts = _lts({"u": "Satisfied", "v": "Violated"})

    assert lts.residual("u", ()) != lts.residual("v", ()), (
        "precondition: the two prefixes must differ on the empty continuation, "
        "otherwise this test proves nothing"
    )

    assert lts.residual("u", ("nope",)) != lts.residual("v", ("nope",)), (
        "the disabled sink erased a distinction the contract had already established"
    )


def test_supplied_continuation_agreement_rejects_distinct_frozen_observations():
    """Different residual encodings disagree for the supplied word.

    This finite result concerns only the supplied continuation. It makes no
    claim about words that the caller did not supply.
    """
    lts = _lts({"u": "Satisfied", "v": "Violated"})

    assert lts.agrees_on_supplied_continuations("u", "v", [("nope",)]) is False, (
        "the supplied continuation produced different frozen observations"
    )


def test_distinction_survives_when_disablement_happens_later_in_the_word():
    """The same requirement when the forbidden label is not the first one."""
    lts = _lts({"u": "Satisfied", "v": "Violated"})

    assert lts.residual("u", ("ok", "ok", "nope")) != lts.residual("v", ("ok", "ok", "nope"))


def test_two_states_agreeing_on_observation_still_compare_equal():
    """Positive control: equal frozen observations still compare equal."""
    lts = _lts({"u": "Satisfied", "v": "Satisfied"})

    assert lts.residual("u", ("nope",)) == lts.residual("v", ("nope",)), (
        "two prefixes the contract does not distinguish must still compare equal after a "
        "disabled label, or the quotient collapses to the identity"
    )
