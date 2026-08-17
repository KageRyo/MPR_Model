from __future__ import annotations

import math

from .settings import CATEGORY_BANDS


def clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def direct_wqi5_score(do: float, bod: float, nh3n: float, ec: float, ss: float) -> float:
    qi_do = (
        -0.08841347
        + (0.8996848 * do)
        - (4.907377e-2 * (do**2))
        + (1.5696e-3 * (do**3))
        - (1.5216e-5 * (do**4))
        + (4.545e-8 * (do**5))
    )
    qi_bod = 1123.6 / (1 + (9.99 * math.exp(0.2 * bod)))
    qi_nh3n = 9.79 + (56.76 / (nh3n + 0.6236888))
    qi_ss = 100.1 - (2.433 * ss) + (2.282e-2 * (ss**2)) - (7.90e-5 * (ss**3))
    qi_ec = 101.7 / (1 + (0.0062 * math.exp(8.32e-3 * ec)))
    return round(sum(clamp_score(q) for q in [qi_do, qi_bod, qi_nh3n, qi_ec, qi_ss]) / 5.0, 3)


def categorize_score(score: float) -> tuple[str, str]:
    score = clamp_score(score)
    for label, lower, upper, rating_range in CATEGORY_BANDS:
        if lower < score <= upper or (lower == 0.0 and lower <= score <= upper):
            return label, rating_range
    return "Unknown", "Out of configured range"


def assess_indicator_quality(indicator: str, value: float) -> str:
    thresholds = {
        "DO": ((80, 120), ((55, 80), (120, 140)), (0, 55)),
        "BOD": ((0, 2), ((2, 4),), (4, 25)),
        "NH3N": ((0, 0.1), ((0.1, 1),), (1, 8)),
        "EC": ((0, 400), ((400, 750),), (750, 3000)),
        "SS": ((0, 10), ((10, 40),), (40, 1000)),
    }
    good, warning_ranges, poor = thresholds[indicator]
    if good[0] <= value <= good[1]:
        return "Good"
    if any(lower <= value <= upper for lower, upper in warning_ranges):
        return "Fair"
    if poor[0] <= value <= poor[1]:
        return "Poor"
    return "OutOfRange"
