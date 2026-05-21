from __future__ import annotations

import random
import threading
from datetime import datetime


SENSOR_MOCK_RANGES = {
    "normal": {
        "temperature": (25, 35),
        "smoke": (20, 100),
        "co": (5, 20),
        "flame": (0, 0),
    },
    "warning": {
        "temperature": (45, 60),
        "smoke": (150, 300),
        "co": (30, 60),
        "flame": (0, 1),
    },
    "fire": {
        "temperature": (60, 90),
        "smoke": (300, 800),
        "co": (60, 150),
        "flame": (1, 1),
    },
}


class SensorMockService:
    def __init__(self) -> None:
        self._mode = "normal"
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def set_mode(self, mode: str) -> dict:
        if mode not in SENSOR_MOCK_RANGES:
            raise ValueError("mode must be one of: normal, warning, fire")
        with self._lock:
            self._mode = mode
        return {
            "success": True,
            "mode": mode,
            "message": f"传感器模拟模式已切换为 {mode}",
        }

    def latest(self) -> dict:
        mode = self.mode
        ranges = SENSOR_MOCK_RANGES[mode]
        flame_low, flame_high = ranges["flame"]
        return {
            "mode": mode,
            "temperature": round(random.uniform(*ranges["temperature"]), 1),
            "smoke": round(random.uniform(*ranges["smoke"]), 1),
            "co": round(random.uniform(*ranges["co"]), 1),
            "flame": random.randint(flame_low, flame_high),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
