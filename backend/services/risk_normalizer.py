from __future__ import annotations


def normalize_visual_risk(fire_confidence: float) -> float:
    if fire_confidence < 0.40:
        return 0.0
    if fire_confidence <= 0.70:
        return 0.5
    return 1.0


def normalize_smoke_risk(smoke: float) -> float:
    if smoke < 150:
        return 0.0
    if smoke <= 300:
        return 0.5
    return 1.0


def normalize_temperature_risk(temperature: float) -> float:
    if temperature < 45:
        return 0.0
    if temperature <= 60:
        return 0.5
    return 1.0


def normalize_co_risk(co: float) -> float:
    if co < 30:
        return 0.0
    if co <= 60:
        return 0.5
    return 1.0


def normalize_flame_risk(flame: int | bool) -> float:
    return 1.0 if int(flame) == 1 else 0.0
