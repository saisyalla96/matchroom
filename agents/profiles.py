"""Viewer profiles: the same match intelligence, genuinely different experiences."""
from __future__ import annotations

from dataclasses import dataclass, field

MODES = ("casual", "analyst", "player_focus", "metric")
METRICS = ("xg", "possession", "pass_accuracy", "pressures", "distance")


@dataclass
class Profile:
    name: str = "viewer"
    mode: str = "casual"            # casual | analyst | player_focus | metric
    language: str = "en"            # BCP-47, e.g. en, es, fr, ar, hi, ja
    clubs: list[str] = field(default_factory=list)   # favourite club ids, e.g. ["NGC"]
    player_id: str | None = None    # for player_focus
    metric: str | None = None       # for metric mode

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.mode == "player_focus" and not self.player_id:
            raise ValueError("player_focus mode needs player_id")
        if self.mode == "metric" and self.metric not in METRICS:
            raise ValueError(f"metric mode needs metric in {METRICS}")


PRESETS = {
    "casual_fan": Profile("Casual fan", "casual"),
    "analyst": Profile("Data analyst", "analyst"),
    "northgate_fan_es": Profile("Northgate fan (Spanish)", "casual", "es", ["NGC"]),
}
