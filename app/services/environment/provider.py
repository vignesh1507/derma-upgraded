"""
Environmental context provider for dermatology — WeatherAPI.com.

Design constraints, in priority order:

1. FAIL OPEN. Supplementary colour, never a precondition for a clinical answer.
   Every failure path returns None and the pipeline carries on.
2. NEVER BLOCK. Short timeout; the request already makes several sequential
   upstream hops before the LLM.
3. CACHE. UV maximum and air quality change on the hour at best.

Uses `forecast.json`, not `current.json`. The daily MAXIMUM UV is the figure a
dermatologist reasons from — current UV is 0.0 after sunset, so an evening
consultation would see no UV signal at all for the most important parameter in
the specialty. One call returns current conditions and the daily aggregates
(UV max, min/max temp) together, so this costs no extra round trip.

Location handling: WeatherAPI resolves bare city names against a global
gazetteer, so `q=Delhi` returns Delhi, ONTARIO, CANADA — a silent, plausible
wrong answer. Indian PIN codes are not supported (HTTP 400).
"""

from __future__ import annotations

import datetime
import re
import threading
import time
from dataclasses import replace
from typing import Any, Callable

import httpx

from app.services.environment.models import (
    DRY_DAY_HUMIDITY_PCT,
    HIGH_UV_DAY_INDEX,
    LOOKBACK_DAYS,
    EnvironmentalContext,
    RecentSkinConditions,
)
from graphrag.utils.logger import get_logger

logger = get_logger(__name__)

_FORECAST_URL = "https://api.weatherapi.com/v1/forecast.json"
_HISTORY_URL = "https://api.weatherapi.com/v1/history.json"

_LATLON_RE = re.compile(r"^\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*$")


class _TTLCache:
    """Small thread-safe TTL cache. Bounded so a hostile input cannot grow it."""

    def __init__(self, ttl_s: float, max_entries: int = 512):
        self._ttl = ttl_s
        self._max = max_entries
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            expires_at, value = hit
            if time.monotonic() >= expires_at:
                self._data.pop(key, None)
                return None
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._data) >= self._max:
                oldest = min(self._data, key=lambda k: self._data[k][0])
                self._data.pop(oldest, None)
            self._data[key] = (time.monotonic() + self._ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def normalise_location(raw: str, default_country: str = "India") -> str:
    """Make a user-supplied location unambiguous."""
    loc = (raw or "").strip()
    if not loc:
        return ""
    if _LATLON_RE.match(loc):
        return loc.replace(" ", "")
    if "," in loc:
        return loc
    return f"{loc}, {default_country}"


# Common short forms callers use versus the full names the gazetteer returns.
# The guard only needs to catch gross mis-resolution (wrong continent), so this
# table is deliberately small rather than a complete ISO mapping.
_COUNTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "uk": ("united kingdom", "great britain", "britain"),
    "usa": ("united states", "united states of america"),
    "us": ("united states", "united states of america"),
    "uae": ("united arab emirates",),
    "india": ("india",),
}


def _country_matches(resolved_label: str, expected: str) -> bool:
    """
    Whether a resolved location sits in the expected country.

    Compares case-insensitively and resolves common abbreviations, because the
    API answers with full names ("United Kingdom") while callers configure
    short ones ("UK") — a naive endswith() discarded correct locations.
    """
    label = (resolved_label or "").strip().lower()
    want = (expected or "").strip().lower()
    if not want:
        return True
    candidates = {want, *_COUNTRY_ALIASES.get(want, ())}
    return any(label.endswith(c) for c in candidates)


def _was_country_assumed(raw: str) -> bool:
    """
    True when we appended the default country because the caller did not
    qualify the location.

    Matters because WeatherAPI's gazetteer is fuzzy: "zzqqxx nowhere" (with
    ", India" appended) resolves to Nowhere, OKLAHOMA, USA, and the service
    would then hand an Indian patient sun advice for the American Midwest. When
    we supplied the country ourselves we must verify the answer honours it;
    when the caller named a country explicitly, we trust them.
    """
    loc = (raw or "").strip()
    return bool(loc) and "," not in loc and not _LATLON_RE.match(loc)


class EnvironmentalProvider:
    """Fetches skin-relevant weather + air quality, cached and fail-open."""

    def __init__(
        self,
        api_key: str | None,
        *,
        timeout_s: float = 2.5,
        cache_ttl_s: float = 3600.0,
        default_country: str = "India",
        lookback_days: int = LOOKBACK_DAYS,
        transport: Callable[[str, dict[str, str], float], dict] | None = None,
    ):
        self._api_key = api_key
        self._timeout = timeout_s
        self._default_country = default_country
        self._lookback_days = max(0, lookback_days)
        self._cache = _TTLCache(cache_ttl_s)
        self._transport = transport or self._http_get

    @staticmethod
    def _http_get(url: str, params: dict[str, str], timeout_s: float) -> dict:
        resp = httpx.get(url, params=params, timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def fetch(self, location: str) -> EnvironmentalContext | None:
        """
        Environmental context for `location`, or None.

        None means "carry on without it" in every case: no key, no location,
        upstream slow, upstream down, payload unexpected.
        """
        if not self.enabled:
            return None

        key = normalise_location(location, self._default_country)
        if not key:
            return None

        cached = self._cache.get(key)
        if cached is not None:
            logger.debug("Environmental cache hit for %s", key)
            return cached

        try:
            payload = self._transport(
                _FORECAST_URL,
                {
                    "key": self._api_key, "q": key,
                    "aqi": "yes", "days": "1", "alerts": "no",
                },
                self._timeout,
            )
        except Exception as exc:                    # noqa: BLE001 - fail open
            logger.warning(
                "Environmental lookup failed for %s (%s: %s) — continuing without it",
                key, type(exc).__name__, str(exc)[:120],
            )
            return None

        ctx = self._parse(payload)
        if ctx is None:
            logger.warning("Environmental payload unusable for %s", key)
            return None

        # Reject a fuzzy match that landed in the wrong country. Only applies
        # when WE supplied the country; an explicit "London, UK" is honoured.
        if _was_country_assumed(location) and not _country_matches(
            ctx.location_label, self._default_country
        ):
            logger.warning(
                "Environmental lookup for %r resolved to %r — outside the "
                "expected country (%s); discarding rather than advising on the "
                "wrong place",
                location, ctx.location_label, self._default_country,
            )
            return None

        # The preceding days are what explain a lesion presenting today.
        # Fetched separately and failing independently: losing the window must
        # never cost us the current reading.
        recent = self._fetch_recent(key)
        if recent is not None:
            ctx = replace(ctx, recent=recent)

        self._cache.put(key, ctx)
        return ctx

    def _fetch_recent(self, key: str) -> RecentSkinConditions | None:
        """Aggregate the last `lookback_days` days. One range call, not N."""
        if self._lookback_days <= 0:
            return None
        today = datetime.date.today()
        start = today - datetime.timedelta(days=self._lookback_days)
        end = today - datetime.timedelta(days=1)
        try:
            payload = self._transport(
                _HISTORY_URL,
                {
                    "key": self._api_key, "q": key,
                    "dt": start.isoformat(), "end_dt": end.isoformat(),
                },
                self._timeout,
            )
        except Exception as exc:                # noqa: BLE001 - fail open
            logger.info(
                "Environmental lookback unavailable for %s (%s) — using today only",
                key, type(exc).__name__,
            )
            return None
        return self._parse_recent(payload)

    @staticmethod
    def _parse_recent(payload: object) -> RecentSkinConditions | None:
        if not isinstance(payload, dict):
            return None
        forecast = payload.get("forecast")
        if not isinstance(forecast, dict):
            return None
        days = forecast.get("forecastday")
        if not isinstance(days, list) or not days:
            return None

        uvs: list[float] = []
        hums: list[int] = []
        maxs: list[float] = []
        mins: list[float] = []
        total_precip = 0.0

        for entry in days:
            day = entry.get("day") if isinstance(entry, dict) else None
            if not isinstance(day, dict):
                continue
            for field, sink in (("uv", uvs), ("maxtemp_c", maxs), ("mintemp_c", mins)):
                try:
                    sink.append(float(day[field]))
                except (KeyError, TypeError, ValueError):
                    pass
            try:
                hums.append(int(day["avghumidity"]))
            except (KeyError, TypeError, ValueError):
                pass
            try:
                total_precip += float(day.get("totalprecip_mm") or 0.0)
            except (TypeError, ValueError):
                pass

        if not uvs:
            # Without UV the window tells a skin service almost nothing.
            return None

        peak_uv = max(uvs)
        return RecentSkinConditions(
            days=len(days),
            peak_uv=peak_uv,
            high_uv_days=sum(1 for u in uvs if u >= HIGH_UV_DAY_INDEX),
            min_avg_humidity_pct=min(hums) if hums else None,
            max_temp_c=max(maxs) if maxs else None,
            min_temp_c=min(mins) if mins else None,
            total_precip_mm=round(total_precip, 1),
        )

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    # ── Parsing ────────────────────────────────────────────────────────────

    @staticmethod
    def _num(source: dict, field: str, default: float = 0.0) -> float:
        try:
            v = source.get(field)
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    @classmethod
    def _parse(cls, payload: object) -> EnvironmentalContext | None:
        if not isinstance(payload, dict):
            return None
        loc = payload.get("location")
        cur = payload.get("current")
        # Present-but-null is a real shape from upstream error envelopes and is
        # distinct from the key being absent — both must fail open.
        if not isinstance(loc, dict) or not isinstance(cur, dict):
            return None

        forecast = payload.get("forecast")
        day: dict = {}
        if isinstance(forecast, dict):
            days = forecast.get("forecastday")
            if isinstance(days, list) and days and isinstance(days[0], dict):
                d0 = days[0].get("day")
                if isinstance(d0, dict):
                    day = d0

        aq = cur.get("air_quality")
        if not isinstance(aq, dict):
            aq = {}

        epa = aq.get("us-epa-index")
        try:
            epa = int(epa) if epa is not None else None
        except (TypeError, ValueError):
            epa = None

        def _opt(source: dict, field: str) -> float | None:
            v = source.get(field)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        label = ", ".join(
            str(loc[k]) for k in ("name", "region", "country") if loc.get(k)
        )

        temp = cls._num(cur, "temp_c")
        # Daily UV max is the headline. Fall back to the current reading only
        # when the forecast block is missing, and accept that it will read 0
        # after dark.
        uv_max = cls._num(day, "uv", cls._num(cur, "uv"))

        try:
            return EnvironmentalContext(
                location_label      = label or "unknown location",
                lat                 = cls._num(loc, "lat"),
                lon                 = cls._num(loc, "lon"),
                observed_local_time = str(loc.get("localtime", "")),
                temp_c              = temp,
                feels_like_c        = cls._num(cur, "feelslike_c", temp),
                humidity_pct        = int(cls._num(cur, "humidity")),
                dewpoint_c          = cls._num(cur, "dewpoint_c"),
                heat_index_c        = cls._num(cur, "heatindex_c", temp),
                condition           = str((cur.get("condition") or {}).get("text", "")),
                precip_mm           = cls._num(cur, "precip_mm"),
                wind_kph            = cls._num(cur, "wind_kph"),
                cloud_pct           = int(cls._num(cur, "cloud")),
                uv_index_max        = uv_max,
                min_temp_c          = cls._num(day, "mintemp_c", temp),
                max_temp_c          = cls._num(day, "maxtemp_c", temp),
                aqi_us_epa          = epa,
                pm2_5               = _opt(aq, "pm2_5"),
                pm10                = _opt(aq, "pm10"),
            )
        except (TypeError, ValueError):
            return None
