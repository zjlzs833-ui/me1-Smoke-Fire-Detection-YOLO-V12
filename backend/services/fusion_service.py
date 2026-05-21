from __future__ import annotations

from backend.config.fusion_weights import FUSION_WEIGHTS
from backend.services.risk_normalizer import (
    normalize_co_risk,
    normalize_flame_risk,
    normalize_smoke_risk,
    normalize_temperature_risk,
    normalize_visual_risk,
)


def calculate_fusion_risk(
    fire_confidence: float,
    temperature: float,
    smoke: float,
    co: float,
    flame: int,
    alarm_threshold: float = 0.70,
) -> dict:
    risk_detail = {
        "visual_risk": normalize_visual_risk(fire_confidence),
        "smoke_risk": normalize_smoke_risk(smoke),
        "temperature_risk": normalize_temperature_risk(temperature),
        "co_risk": normalize_co_risk(co),
        "flame_risk": normalize_flame_risk(flame),
    }
    risk_score = (
        FUSION_WEIGHTS["visual"] * risk_detail["visual_risk"]
        + FUSION_WEIGHTS["smoke"] * risk_detail["smoke_risk"]
        + FUSION_WEIGHTS["temperature"] * risk_detail["temperature_risk"]
        + FUSION_WEIGHTS["co"] * risk_detail["co_risk"]
        + FUSION_WEIGHTS["flame"] * risk_detail["flame_risk"]
    )
    risk_score = round(risk_score, 4)

    if risk_score < 0.4:
        risk_level = "正常"
        alert_level = "safe"
        alarm = False
    elif risk_score < alarm_threshold:
        risk_level = "预警"
        alert_level = "warning"
        alarm = False
    else:
        risk_level = "火灾报警"
        alert_level = "danger"
        alarm = True

    return {
        "fire_confidence": round(float(fire_confidence), 4),
        "sensor_data": {
            "temperature": round(float(temperature), 1),
            "smoke": round(float(smoke), 1),
            "co": round(float(co), 1),
            "flame": int(flame),
        },
        "risk_detail": risk_detail,
        "weight_detail": FUSION_WEIGHTS.copy(),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "alert_level": alert_level,
        "alarm": alarm,
        "alarm_threshold": round(float(alarm_threshold), 2),
    }
