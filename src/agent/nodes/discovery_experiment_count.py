"""
src/agent/nodes/discovery_experiment_count.py
"""

import json
import math

from src.agent.nodes.discovery_experiment import RankField


def _default_stage_counts(n: int, pool_size: int) -> list[int]:
    """
    Returns n default counts, one per stage, outer to inner. Each
    value is half of the previous one (rounded up), starting from
    pool_size. Scales with the candidate pool, so a small pool (e.g.
    a niche industry search) gets small defaults instead of numbers
    bigger than the pool itself.
    """
    defaults = []
    current = pool_size
    for _ in range(n):
        current = math.ceil(current / 2)
        defaults.append(current)
    return defaults


def _apply_defaults(stages: list[list[RankField]], defaults: list[int]) -> list[list[RankField]]:
    """
    Writes each default count onto every field in each stage, so a
    multi-field (averaged) stage has the same count on all its
    fields, not just one — no null count sitting next to a real one.
    """
    new_stages = []
    for stage, default_count in zip(stages, defaults):
        new_stage = [field.model_copy(update={"count": default_count}) for field in stage]
        new_stages.append(new_stage)
    return new_stages


def _trailing_nones_only(counts: list[int | None]) -> bool:
    """
    True if every None in counts is at the tail — i.e. once a real
    number appears, no None follows it further right. False if any
    None has a real number somewhere to its right (a None "sandwiched"
    between two known numbers, however far apart).
    """
    last_number_index = max(i for i, c in enumerate(counts) if c is not None)
    return all(c is not None for c in counts[:last_number_index + 1])


def _fill_trailing(counts: list[int | None]) -> list[int]:
    """
    Fills trailing Nones (nothing but None after the last given
    number) by halving the last known number repeatedly, one halving
    per remaining stage.
    """
    last_number_index = max(i for i, c in enumerate(counts) if c is not None)
    result = list(counts)
    current = counts[last_number_index]
    for i in range(last_number_index + 1, len(counts)):
        current = math.ceil(current / 2)
        result[i] = current
    return result


def _fill_sandwiched(counts: list[int | None], pool_size: int) -> list[int]:
    """
    Fills Nones sandwiched between two known numbers (or between
    pool_size and the first known number) with a linear interpolation,
    rounded up at each step. Handles multiple separate sandwiched
    stretches, and a trailing stretch (no right-hand anchor) by
    halving from the last known number onward.

    Example: pool_size=10, counts=[None, 5, None, None]
        -> anchors at position -1 (pool_size=10) and position 1 (5):
           position 0 gets ceil(10 - (10-5)/2) = 8
        -> trailing Nones after position 1 (value 5) get halved:
           position 2 = ceil(5/2) = 3, position 3 = ceil(3/2) = 2
        -> result: [8, 5, 3, 2]
    """
    full = [pool_size] + list(counts)  # pool_size is the anchor at index 0

    known_indices = [i for i, c in enumerate(full) if c is not None]

    for left_idx, right_idx in zip(known_indices, known_indices[1:]):
        left_value = full[left_idx]
        right_value = full[right_idx]
        steps = right_idx - left_idx
        for i in range(left_idx + 1, right_idx):
            fraction = (i - left_idx) / steps
            full[i] = math.ceil(left_value - fraction * (left_value - right_value))

    last_known_idx = known_indices[-1]
    if last_known_idx < len(full) - 1:
        current = full[last_known_idx]
        for i in range(last_known_idx + 1, len(full)):
            current = math.ceil(current / 2)
            full[i] = current

    return full[1:]  # drop the pool_size anchor, return only the stage counts


def determine_stage_counts(stages: list[list[RankField]], pool_size: int) -> list[list[RankField]]:
    """
    Takes group_fields_by_priority's output (already split into ordered
    stages, outer to inner) and the size of the candidate pool this
    query starts from.

    Same input/output shape: a list of stages in, a list of stages
    out (same length, same field order) — only the counts inside may
    be corrected.

    Classification:
      1. Not self-consistent -> defaults.
      2. Self-consistent:
         a. Exceeds pool_size -> defaults.
         b. Within pool_size:
            i.  No None anywhere -> return as-is.
            ii. Contains at least one None:
                - All None -> defaults.
                - Mixed (at least one real number given):
                    - All Nones trailing (nothing but None after
                      the last given number) -> halve the last
                      given number repeatedly for the remaining
                      stages.
                    - A None sandwiched between two known numbers
                      -> linear interpolation between the
                      surrounding anchors (pool_size counts as the
                      anchor before the first stage), rounded up.
    """
    counts = []
    for stage in stages:
        user_count = next((f.count for f in stage if f.count is not None), None)
        counts.append(user_count)

    given = [c for c in counts if c is not None]
    self_consistent = all(given[i] >= given[i + 1] for i in range(len(given) - 1))

    print(f"  [determine_stage_counts] pool_size={pool_size}, counts={counts}, given={given}")

    if not self_consistent:
        # Branch 1: not self-consistent — defaults.
        print(f"  [determine_stage_counts] not self-consistent")
        result_stages = _apply_defaults(stages, _default_stage_counts(len(stages), pool_size))

    else:
        # Branch 2: self-consistent.
        exceeds_pool = given and given[0] > pool_size

        if exceeds_pool:
            # Branch 2a: exceeds pool_size — defaults.
            print(f"  [determine_stage_counts] outer count exceeds pool_size ({pool_size})")
            result_stages = _apply_defaults(stages, _default_stage_counts(len(stages), pool_size))

        else:
            # Branch 2b: within pool_size.
            has_none = any(c is None for c in counts)

            if not has_none:
                # Branch 2b-i: every stage has a count, but stages
                # with multiple fields (averaged) may only have it on
                # one field — broadcast to keep all fields in a stage
                # consistent, same as the other branches.
                print(f"  [determine_stage_counts] valid, within pool_size ({pool_size})")
                result_stages = _apply_defaults(stages, counts)

            else:
                # Branch 2b-ii: contains at least one None.
                all_none = all(c is None for c in counts)

                if all_none:
                    # All None — defaults.
                    print(f"  [determine_stage_counts] all None")
                    result_stages = _apply_defaults(stages, _default_stage_counts(len(stages), pool_size))

                else:
                    # Mixed: at least one real number given.
                    trailing_only = _trailing_nones_only(counts)

                    if trailing_only:
                        # All Nones trailing — halve the last given number onward.
                        print(f"  [determine_stage_counts] trailing None(s), halved forward")
                        result_stages = _apply_defaults(stages, _fill_trailing(counts))

                    else:
                        # A None sandwiched between two known numbers — interpolate.
                        print(f"  [determine_stage_counts] None sandwiched, interpolated")
                        result_stages = _apply_defaults(stages, _fill_sandwiched(counts, pool_size))

    stages_as_json = [[f.model_dump() for f in stage] for stage in result_stages]
    print(json.dumps(stages_as_json, indent=2))

    return result_stages
