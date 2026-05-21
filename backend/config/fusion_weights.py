from __future__ import annotations

FUSION_WEIGHTS = {
    "visual": 0.50,
    "smoke": 0.20,
    "temperature": 0.15,
    "co": 0.10,
    "flame": 0.05,
}


def validate_fusion_weights() -> None:
    total = round(sum(FUSION_WEIGHTS.values()), 6)
    if total != 1.0:
        raise ValueError(f"Fusion weights must sum to 1.0, got {total}")
