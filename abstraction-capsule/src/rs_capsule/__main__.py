from __future__ import annotations

import dataclasses
import json
import sys
from enum import Enum
from typing import Any

from rs_metalang_ref.contracts import (
    Breach,
    Broadcast,
    Indeterminate,
    Linear,
    WaiveIf,
)

from .capsule import SOURCE_TRACE, CapsuleResult, run_capsule
from .cards import open_obligation_card, truncation_card
from .profile import build_profile

_TAGGED_CONTRACT_VARIANTS = (Linear, Broadcast, Breach, Indeterminate, WaiveIf)


def _json_value(value: Any) -> Any:
    if isinstance(value, _TAGGED_CONTRACT_VARIANTS):
        return {
            type(value).__name__: {
                field.name: _json_value(getattr(value, field.name))
                for field in dataclasses.fields(value)
            }
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_value(item) for item in value)
    return value


def _result_record(result: CapsuleResult) -> dict[str, Any]:
    comparison = result.comparison
    return {
        "profile_id": result.profile_id,
        "card_id": result.card_id,
        "comparison": {type(comparison).__name__: _json_value(comparison)},
        "run_metadata": _json_value(result.run_metadata),
        "summary": result.summary,
    }


def main() -> int:
    if sys.argv[1:]:
        print("error: rs_capsule does not accept arguments", file=sys.stderr)
        return 2

    profile = build_profile()
    results = (
        run_capsule(profile, truncation_card(1), SOURCE_TRACE),
        run_capsule(profile, open_obligation_card(), SOURCE_TRACE),
    )
    print(json.dumps([_result_record(result) for result in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
