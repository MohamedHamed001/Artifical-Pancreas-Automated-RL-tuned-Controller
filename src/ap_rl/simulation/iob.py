"""Research-only rapid-acting insulin-on-board accounting.

The discrete curve is the remaining mass of two equal first-order action
stages (an Erlang-2 model), sampled once per minute. Its time constant is one
quarter of the configured duration, so the default 240-minute curve has peak
action at about 60 minutes. The curve is normalized to reach exactly zero at
the configured cutoff. This is a transparent simulation assumption, not a
clinically validated dosing model.
"""

from math import exp

DEFAULT_IOB_DURATION_MIN = 240


class RapidActingIOB:
    def __init__(self, duration_min: int = DEFAULT_IOB_DURATION_MIN) -> None:
        if (
            isinstance(duration_min, bool)
            or not isinstance(duration_min, int)
            or duration_min <= 0
        ):
            raise ValueError("duration_min must be a positive whole number of minutes")
        self.duration_min = duration_min
        self._time_constant_min = duration_min / 4.0
        self._cutoff_remaining = 5.0 * exp(-4.0)
        self._doses: list[tuple[int, float]] = []

    def advance(self, delivered_u: float) -> float:
        if delivered_u < 0:
            raise ValueError("delivered_u must be nonnegative")
        self._doses = [
            (age_min + 1, dose_u)
            for age_min, dose_u in self._doses
            if age_min + 1 < self.duration_min
        ]
        if delivered_u:
            self._doses.append((0, delivered_u))
        return self.iob_u

    def reset(self) -> None:
        self._doses.clear()

    @property
    def iob_u(self) -> float:
        return sum(
            dose_u * self._remaining_fraction(age_min)
            for age_min, dose_u in self._doses
        )

    def _remaining_fraction(self, age_min: int) -> float:
        scaled_age = age_min / self._time_constant_min
        unscaled = exp(-scaled_age) * (1.0 + scaled_age)
        return (unscaled - self._cutoff_remaining) / (
            1.0 - self._cutoff_remaining
        )
