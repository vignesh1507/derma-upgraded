"""
Typed environmental context for a dermatology service.

Skin responds to ABSOLUTE conditions, not to local deviation from normal. A dew
point of 0.9 C strips the barrier and causes xerosis whether or not Leh is
always like that; UV 14 burns the same everywhere. Every threshold below is
therefore physiological and applied uniformly across India rather than
calibrated to any one city — sampling eleven climate zones on one day gave
UV 7.8-14.1, humidity 33-98%, dew point 0.9-24.4 C, so a Delhi-tuned band set
would have been wrong nearly everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# UV — WHO Global Solar UV Index. Standardised worldwide, so it needs no
# regional adjustment. This is the single most important dermatological
# parameter and the reason derma reads the DAILY MAXIMUM rather than the
# current value: current UV is 0 after sunset, and an evening consultation
# would otherwise conclude sun exposure is irrelevant.
# ---------------------------------------------------------------------------
_UV_BANDS: list[tuple[float, str, str]] = [
    (11.0, "Extreme",
     "Unprotected skin can burn within minutes; shade and covering are the "
     "priority, sunscreen alone is not enough."),
    (8.0, "Very high",
     "Significant burn risk; midday sun is best avoided."),
    (6.0, "High",
     "Sun protection needed for outdoor activity."),
    (3.0, "Moderate",
     "Matters for photosensitive conditions and photosensitising drugs."),
    (0.0, "Low", ""),
]

# At or above this, UV is worth raising for anyone. Below it, UV still matters
# for photosensitive disease, which the prompt layer handles.
UV_RELEVANT_INDEX: float = 6.0
UV_PHOTOSENSITIVE_INDEX: float = 3.0

# ---------------------------------------------------------------------------
# Moisture. Dew point beats relative humidity for skin: Kochi at 86% RH and
# Bikaner at 40% RH look opposite, but their dew points (23.0 and 21.0) are
# close and both mean sweat cannot evaporate. RH without temperature misleads.
# ---------------------------------------------------------------------------
VERY_DRY_DEWPOINT_C: float = 5.0     # xerosis, eczema flares, barrier loss
OCCLUSIVE_DEWPOINT_C: float = 20.0   # miliaria, intertrigo, tinea, candidiasis
HEAT_INDEX_SWEAT_C: float = 35.0     # hyperhidrosis, prickly heat, acne
CHILBLAIN_MIN_TEMP_C: float = 10.0   # chilblains, cold urticaria, Raynaud's

# Particulates aggravate atopic dermatitis, acne and urticaria. Reusing the US
# EPA index the provider already returns rather than inventing a skin scale.
PM_RELEVANT_EPA_BAND: int = 3

# ---------------------------------------------------------------------------
# Lookback. Skin lags exposure harder than any other specialty:
#     sunburn              onset 4-24 h, PEAKS at 24-72 h
#     contact dermatitis   type IV hypersensitivity, 24-72 h
#     photodermatoses      typically next day
#     xerosis / eczema     cumulative over a dry spell
#     miliaria             builds over days of sustained sweating
#
# A patient presenting today with a burnt face was burnt YESTERDAY. Today's UV
# describes the exposure that has not yet produced symptoms and says nothing
# about the one that did — Leh read UV 15.6 seven days ago against 5.3 three
# days ago, so a single-day reading badly misrepresents the week.
#
# Seven days covers both lag profiles: the acute 1-3 day events and the
# cumulative barrier stress that needs longer.
LOOKBACK_DAYS: int = 7

# A day at or above this counts as a high-UV exposure day.
HIGH_UV_DAY_INDEX: float = 8.0
# Mean relative humidity at or below this marks a genuinely dry day. Used as
# the window's dryness signal because the daily aggregate exposes humidity but
# not dew point; deriving dew point from daily means would be inventing a
# number the API did not give us.
DRY_DAY_HUMIDITY_PCT: int = 35

# Appended to every block. Derma sees more "always true" conditions than a
# chest service — Shillong sits at 98% humidity permanently — so the risk is
# the model treating an ambient reading as this patient's diagnosis.
_FRAMING_LINE: str = (
    "Background, not a conclusion. Ambient temperature and humidity are "
    "unavoidable here, indoors too — state them as fact rather than asking "
    "whether the air is dry or cold. UV and particulate exposure depend on "
    "behaviour and protection, so ask before attributing. See EXPOSURE "
    "HISTORY."
)


@dataclass(frozen=True)
class RecentSkinConditions:
    """
    Aggregate of the days preceding today — the exposure window that actually
    explains a lesion presenting now.

    Every field comes straight from a daily aggregate the API returns. Nothing
    is derived, so nothing can be silently wrong.
    """

    days: int
    peak_uv: float
    high_uv_days: int          # days at or above HIGH_UV_DAY_INDEX
    min_avg_humidity_pct: int | None = None
    max_temp_c: float | None = None
    min_temp_c: float | None = None
    total_precip_mm: float = 0.0

    @property
    def had_high_uv(self) -> bool:
        return self.peak_uv >= HIGH_UV_DAY_INDEX

    @property
    def had_dry_spell(self) -> bool:
        return (
            self.min_avg_humidity_pct is not None
            and self.min_avg_humidity_pct <= DRY_DAY_HUMIDITY_PCT
        )

    @property
    def is_noteworthy(self) -> bool:
        return self.had_high_uv or self.had_dry_spell

    def summary(self) -> str:
        bits: list[str] = []
        if self.had_high_uv:
            bits.append(
                f"peak UV {self.peak_uv:.0f} on {self.high_uv_days} of the "
                f"last {self.days} days"
            )
        if self.had_dry_spell:
            bits.append(f"humidity down to {self.min_avg_humidity_pct}%")
        if not bits:
            return ""
        return (
            "Recent (skin lesions often lag exposure by 1-3 days): "
            + ", ".join(bits)
        )


@dataclass(frozen=True)
class EnvironmentalContext:
    """Environmental reading for one location, framed for skin."""

    location_label: str
    lat: float
    lon: float
    observed_local_time: str

    # Weather — current
    temp_c: float
    feels_like_c: float
    humidity_pct: int
    dewpoint_c: float
    heat_index_c: float
    condition: str
    precip_mm: float
    wind_kph: float
    cloud_pct: int

    # Daily aggregates — UV max is the headline for a skin service
    uv_index_max: float
    min_temp_c: float
    max_temp_c: float

    # Air quality
    aqi_us_epa: int | None = None
    pm2_5: float | None = None
    pm10: float | None = None

    # Preceding days — None when the lookback is disabled or unavailable.
    recent: "RecentSkinConditions | None" = None

    # ── UV ─────────────────────────────────────────────────────────────────

    @property
    def uv_band(self) -> str:
        for threshold, label, _ in _UV_BANDS:
            if self.uv_index_max >= threshold:
                return label
        return "Low"

    @property
    def uv_note(self) -> str:
        for threshold, _, note in _UV_BANDS:
            if self.uv_index_max >= threshold:
                return note
        return ""

    # ── Moisture / temperature ─────────────────────────────────────────────

    @property
    def moisture_note(self) -> str:
        """
        One line on what the air is doing to the skin barrier.

        Deliberately describes the mechanism, not a diagnosis — the model is
        told elsewhere never to convert this into a cause.
        """
        if self.dewpoint_c <= VERY_DRY_DEWPOINT_C:
            return ("Air is very dry — accelerates water loss from the skin "
                    "barrier.")
        if self.dewpoint_c >= OCCLUSIVE_DEWPOINT_C:
            return ("Air is humid enough that sweat evaporates poorly — skin "
                    "folds stay damp.")
        return ""

    @property
    def thermal_note(self) -> str:
        if self.heat_index_c >= HEAT_INDEX_SWEAT_C:
            return "Heat stress high — sustained sweating likely."
        if self.min_temp_c <= CHILBLAIN_MIN_TEMP_C:
            return "Cold overnight — relevant to cold-triggered skin problems."
        return ""

    # ── Gating ─────────────────────────────────────────────────────────────

    @property
    def is_clinically_relevant(self) -> bool:
        """
        Whether this reading earns its place in the prompt.

        Deliberately more permissive than the chest service: UV and barrier
        conditions change dermatological advice even when the presenting
        complaint is something else, because sun protection and emollient
        advice are things a dermatologist would give anyway.
        """
        if self.uv_index_max >= UV_RELEVANT_INDEX:
            return True
        if self.dewpoint_c <= VERY_DRY_DEWPOINT_C:
            return True
        if self.dewpoint_c >= OCCLUSIVE_DEWPOINT_C:
            return True
        if self.heat_index_c >= HEAT_INDEX_SWEAT_C:
            return True
        if self.min_temp_c <= CHILBLAIN_MIN_TEMP_C:
            return True
        if self.aqi_us_epa is not None and self.aqi_us_epa >= PM_RELEVANT_EPA_BAND:
            return True
        # A mild day after a scorching or parched week still explains today's
        # lesion. This is the whole point of the lookback.
        if self.recent is not None and self.recent.is_noteworthy:
            return True
        return False

    # ── Rendering ──────────────────────────────────────────────────────────

    def to_prompt_block(self) -> str:
        lines = [
            f"Location: {self.location_label} (local time {self.observed_local_time})",
            f"UV index today (max): {self.uv_index_max:.0f} — {self.uv_band}",
        ]
        if self.uv_note:
            lines.append(self.uv_note)

        lines.append(
            f"Weather: {self.temp_c:.0f}C (feels {self.feels_like_c:.0f}C), "
            f"humidity {self.humidity_pct}%, dew point {self.dewpoint_c:.0f}C, "
            f"{self.condition.lower()}"
        )
        if self.moisture_note:
            lines.append(self.moisture_note)
        if self.thermal_note:
            lines.append(self.thermal_note)

        if self.aqi_us_epa is not None and self.aqi_us_epa >= PM_RELEVANT_EPA_BAND:
            pm = f"Air quality: US EPA band {self.aqi_us_epa} of 6"
            if self.pm2_5 is not None:
                pm += f", PM2.5 {self.pm2_5:.0f} ug/m3"
            lines.append(pm)
            lines.append(
                "Particulate load is a recognised aggravator of atopic "
                "dermatitis, acne and urticaria."
            )

        if self.recent is not None:
            recent_line = self.recent.summary()
            if recent_line:
                lines.append(recent_line)
        lines.append(_FRAMING_LINE)
        return "[LOCAL CONDITIONS]\n" + "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location_label,
            "lat": self.lat,
            "lon": self.lon,
            "observed_local_time": self.observed_local_time,
            "uv_index_max": self.uv_index_max,
            "uv_band": self.uv_band,
            "temp_c": self.temp_c,
            "feels_like_c": self.feels_like_c,
            "humidity_pct": self.humidity_pct,
            "dewpoint_c": self.dewpoint_c,
            "heat_index_c": self.heat_index_c,
            "min_temp_c": self.min_temp_c,
            "max_temp_c": self.max_temp_c,
            "condition": self.condition,
            "precip_mm": self.precip_mm,
            "wind_kph": self.wind_kph,
            "cloud_pct": self.cloud_pct,
            "aqi_us_epa": self.aqi_us_epa,
            "pm2_5": self.pm2_5,
            "pm10": self.pm10,
            "clinically_relevant": self.is_clinically_relevant,
            "recent": (
                {
                    "days": self.recent.days,
                    "peak_uv": self.recent.peak_uv,
                    "high_uv_days": self.recent.high_uv_days,
                    "min_avg_humidity_pct": self.recent.min_avg_humidity_pct,
                    "had_high_uv": self.recent.had_high_uv,
                    "had_dry_spell": self.recent.had_dry_spell,
                }
                if self.recent else None
            ),
        }
