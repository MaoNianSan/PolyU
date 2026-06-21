from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Unit:
    name: str
    group: str
    phrases: tuple[str, ...]
    critical: bool = False

OLD10 = [
    Unit("boy_child", "object", ("boy", "child", "little boy", "son"), True),
    Unit("girl_daughter", "object", ("girl", "daughter", "sister", "little girl"), True),
    Unit("mother_woman", "object", ("mother", "woman", "lady", "mom"), True),
    Unit("cookie_jar", "object", ("cookie jar", "cookies", "cookie", "jar"), True),
    Unit("stool_chair_falling", "hazard", ("stool", "chair", "fall", "falling", "tip over"), True),
    Unit("sink_water_overflow", "hazard", ("sink", "water", "overflow", "running water", "spilling"), True),
    Unit("dish_washing", "action", ("dish", "dishes", "washing", "drying dish", "plate")),
    Unit("kitchen", "scene", ("kitchen",)),
    Unit("window_curtain", "scene", ("window", "curtain", "curtains")),
    Unit("taking_reaching_stealing", "action", ("take", "taking", "reach", "reaching", "steal", "stealing"), True),
]

EXPANDED = [
    Unit("boy", "object", ("boy", "little boy", "son", "child"), True),
    Unit("girl", "object", ("girl", "little girl", "daughter", "sister"), True),
    Unit("mother", "object", ("mother", "mom", "woman", "lady"), True),
    Unit("cookie", "object", ("cookie", "cookies"), True),
    Unit("cookie_jar", "object", ("cookie jar", "jar", "cookies in jar"), True),
    Unit("stool", "object", ("stool", "chair", "step stool"), True),
    Unit("sink", "object", ("sink", "basin"), True),
    Unit("water", "object", ("water", "faucet", "tap"), True),
    Unit("dish", "object", ("dish", "dishes", "plate")),
    Unit("window", "object", ("window", "windows")),
    Unit("curtain", "object", ("curtain", "curtains")),
    Unit("boy_taking_cookie", "action", ("boy taking cookie", "boy take cookie", "steal cookie", "taking cookies", "get cookie", "reaching cookie"), True),
    Unit("girl_reaching_cookie", "action", ("girl reaching", "girl wants cookie", "girl waiting", "sister waiting", "hand up")),
    Unit("mother_washing_dishes", "action", ("mother washing dishes", "woman washing dishes", "drying dish", "washing dish", "doing dishes"), True),
    Unit("water_overflowing", "action", ("water overflowing", "water running", "sink overflowing", "tap on", "faucet on", "water spilling"), True),
    Unit("stool_falling", "action", ("stool falling", "chair falling", "fall off", "tip over", "tipping"), True),
    Unit("child_climbing", "action", ("boy climbing", "child climbing", "standing on stool", "on stool"), True),
    Unit("stool_tipping", "hazard", ("stool tipping", "tip over", "falling stool", "lose balance"), True),
    Unit("water_spilling", "hazard", ("water spilling", "water overflowing", "spill", "overflow"), True),
    Unit("child_fall_risk", "hazard", ("fall", "falling", "hurt himself", "accident", "danger"), True),
    Unit("mother_unaware", "hazard", ("mother unaware", "not looking", "oblivious", "doesn't see", "not paying attention"), True),
    Unit("sink_overflow_risk", "hazard", ("sink overflow", "water over the sink", "water on floor"), True),
    Unit("kitchen_scene", "scene", ("kitchen", "kitchen scene")),
    Unit("two_children", "scene", ("two children", "boy and girl", "brother and sister"), True),
    Unit("mother_at_sink", "scene", ("mother at sink", "woman at sink", "standing by sink"), True),
    Unit("cookie_theft_scene", "scene", ("cookie theft", "taking cookies", "stealing cookies", "cookie jar"), True),
]

RELATION = [
    Unit("boy_on_stool", "relation", ("boy on stool", "standing on stool", "boy standing on chair", "child on stool"), True),
    Unit("boy_reaching_cookie_jar", "relation", ("boy reaching cookie jar", "boy getting cookies", "boy taking cookies", "reaching into cookie jar"), True),
    Unit("girl_waiting_for_cookie", "relation", ("girl waiting for cookie", "girl wants cookie", "girl asking for cookie", "hand out")),
    Unit("mother_at_sink_relation", "relation", ("mother at sink", "woman by sink", "mother washing dishes"), True),
    Unit("mother_unaware_of_children", "relation", ("mother not looking", "mother unaware", "doesn't see children", "oblivious to children"), True),
    Unit("sink_water_overflowing", "relation", ("sink overflowing", "water running over", "water spilling from sink", "faucet running"), True),
    Unit("stool_fall_risk", "relation", ("stool tipping", "boy may fall", "fall off stool", "chair tipping"), True),
    Unit("child_accident_risk", "relation", ("child could fall", "boy could get hurt", "dangerous", "accident"), True),
]

def unique_units(units: list[Unit]) -> list[Unit]:
    seen = set()
    out = []
    for u in units:
        if u.name not in seen:
            out.append(u); seen.add(u.name)
    return out

def get_units(version: str) -> list[Unit]:
    """Return information-unit definitions for legacy and refined experiments."""
    if version == "early_v0_old10":
        return OLD10
    if version == "early_v1_expanded":
        return EXPANDED
    if version == "early_v2_soft_bm25":
        return EXPANDED
    if version == "early_v3_coverage_omission":
        return unique_units(EXPANDED + RELATION)
    if version == "early_v4_relation":
        return RELATION
    if version == "early_v5_all":
        return unique_units(OLD10 + EXPANDED + RELATION)

    # Refined V5 experiment versions
    if version == "early_v5_full":
        return unique_units(OLD10 + EXPANDED + RELATION)
    if version == "early_v5_no_relation":
        return unique_units(OLD10 + EXPANDED)
    if version == "early_v5_relation_only":
        return RELATION
    if version == "early_v5_coverage_omission_only":
        return unique_units(OLD10 + EXPANDED + RELATION)
    if version == "early_v5_balanced_refined":
        return unique_units(OLD10 + EXPANDED)
    if version == "early_v5_mild_sensitive":
        return unique_units(OLD10 + EXPANDED + RELATION)
    raise ValueError(f"Unknown feature version: {version}")

def relation_units() -> list[Unit]:
    return RELATION
