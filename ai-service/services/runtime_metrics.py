import time
from collections import deque
from typing import Deque, Dict

_START_TIME = time.time()
_LATENCIES_MS: Deque[float] = deque(maxlen=10)


def record_latency_ms(latency_ms: float) -> None:
    _LATENCIES_MS.append(float(latency_ms))


def get_avg_latency_ms() -> float:
    if not _LATENCIES_MS:
        return 0.0
    return round(sum(_LATENCIES_MS) / len(_LATENCIES_MS), 2)


def get_latency_sample_count() -> int:
    return len(_LATENCIES_MS)


def get_uptime_seconds() -> int:
    return int(time.time() - _START_TIME)


def get_uptime_human() -> str:
    total = get_uptime_seconds()
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_runtime_stats() -> Dict:
    return {
        "uptime_seconds": get_uptime_seconds(),
        "uptime_human": get_uptime_human(),
        "avg_response_time_ms_last_10": get_avg_latency_ms(),
        "response_time_samples": get_latency_sample_count(),
    }
