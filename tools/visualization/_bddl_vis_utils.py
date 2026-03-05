#!/usr/bin/env python3


def sanitize_ranges(raw_ranges, require_non_empty=True):
    cleaned = []
    for entry in raw_ranges:
        if len(entry) != 4:
            raise ValueError(f"Expected 4 values per range, received {entry}")
        x0, y0, x1, y1 = entry
        cleaned.append(
            (
                min(float(x0), float(x1)),
                min(float(y0), float(y1)),
                max(float(x0), float(x1)),
                max(float(y0), float(y1)),
            )
        )
    if require_non_empty and not cleaned:
        raise ValueError("No ranges defined for the requested region.")
    return cleaned


def infer_region_key(initial_state, object_name):
    for state in initial_state:
        if (
            isinstance(state, list)
            and len(state) >= 3
            and state[0].lower() == "on"
            and state[1] == object_name
        ):
            return state[2]
    raise ValueError(
        f"Could not infer region for {object_name} from initial_state definitions."
    )
