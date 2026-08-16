"""
EnvironmentalProvider (dermatology) — UV banding, moisture, fail-open, caching.

No test touches the network: the HTTP call is injected.

Governing requirement: this provider can NEVER degrade a clinical answer. Every
failure mode returns None quietly, asserted individually rather than trusting a
blanket try/except.

Thresholds are absolute physiology, not Delhi-calibrated. Sampling eleven Indian
climate zones on one day gave UV 7.8-14.1, humidity 33-98%, dew point 0.9-24.4 C
— a band set tuned to any single city would be wrong nearly everywhere.
"""

from __future__ import annotations

import httpx
import pytest

from app.services.environment import (
    EnvironmentalContext,
    EnvironmentalProvider,
    normalise_location,
)


def _payload(*, uv_max=7.8, dewpoint=24.0, heatindex=38.4, mintemp=30.0,
             humidity=62, epa=4, temp=32.0, condition="Light rain shower"):
    return {
        "location": {
            "name": "Delhi", "region": "Delhi", "country": "India",
            "lat": 28.65, "lon": 77.22, "localtime": "2026-08-10 20:23",
        },
        "current": {
            "temp_c": temp, "feelslike_c": 38.0, "humidity": humidity,
            "dewpoint_c": dewpoint, "heatindex_c": heatindex,
            "condition": {"text": condition}, "precip_mm": 0.1,
            "wind_kph": 10.4, "cloud": 88, "uv": 0.0,
            "air_quality": {"pm2_5": 77.8, "pm10": 64.5, "us-epa-index": epa},
        },
        "forecast": {"forecastday": [{
            "date": "2026-08-10",
            "day": {"uv": uv_max, "mintemp_c": mintemp, "maxtemp_c": 36.2,
                    "avghumidity": humidity},
        }]},
    }


def _history(uvs, humidities=None, maxtemps=None, precips=None):
    """WeatherAPI history.json range payload built from per-day UV values."""
    n = len(uvs)
    humidities = humidities or [60] * n
    maxtemps = maxtemps or [30.0] * n
    precips = precips or [0.0] * n
    return {"forecast": {"forecastday": [
        {"date": f"2026-08-0{i+1}",
         "day": {"uv": uvs[i], "avghumidity": humidities[i],
                 "maxtemp_c": maxtemps[i], "mintemp_c": 20.0,
                 "totalprecip_mm": precips[i]}}
        for i in range(n)
    ]}}


def _routing(current=None, history=None):
    """Transport answering forecast.json and history.json differently."""
    def _t(url, params, timeout):
        if "history" in url:
            if history is None:
                raise httpx.HTTPStatusError(
                    "no history", request=httpx.Request("GET", url),
                    response=httpx.Response(403))
            return history
        return current if current is not None else _payload()
    return _t


def _provider(transport=None, **kw):
    kw.setdefault("api_key", "test-key")
    kw.setdefault("lookback_days", 0)      # most tests exercise today-only
    return EnvironmentalProvider(
        transport=transport or (lambda u, p, t: _payload()), **kw)


# ---------------------------------------------------------------------------
# UV — the headline parameter, and the reason we read forecast not current
# ---------------------------------------------------------------------------

class TestUVIndex:
    def test_uses_daily_max_not_current(self):
        """
        Current UV reads 0.0 after sunset. Using it would make every evening
        consultation conclude sun exposure is irrelevant — for the single most
        important parameter in dermatology.
        """
        ctx = _provider(transport=lambda u, p, t: _payload(uv_max=11.5)).fetch("Delhi")
        assert ctx.uv_index_max == 11.5, "must read forecast day.uv, not current.uv"

    @pytest.mark.parametrize("uv,band", [
        (0.5, "Low"), (2.0, "Low"),
        (3.0, "Moderate"), (5.9, "Moderate"),
        (6.0, "High"), (7.8, "High"),
        (8.0, "Very high"), (10.6, "Very high"),
        (11.0, "Extreme"), (14.1, "Extreme"),
    ])
    def test_who_bands(self, uv, band):
        """WHO Global Solar UV Index — standardised, so no regional tuning."""
        ctx = _provider(transport=lambda u, p, t: _payload(uv_max=uv)).fetch("X")
        assert ctx.uv_band == band

    def test_real_indian_extremes_land_in_the_right_bands(self):
        """Leh 14.1 (extreme, high altitude) vs Delhi 7.8 (high)."""
        leh = _provider(transport=lambda u, p, t: _payload(uv_max=14.1)).fetch("Leh")
        delhi = _provider(transport=lambda u, p, t: _payload(uv_max=7.8)).fetch("Delhi")
        assert leh.uv_band == "Extreme" and delhi.uv_band == "High"

    def test_extreme_uv_says_shade_not_just_sunscreen(self):
        ctx = _provider(transport=lambda u, p, t: _payload(uv_max=14.1)).fetch("Leh")
        assert "sunscreen alone is not enough" in ctx.uv_note

    def test_low_uv_carries_no_note(self):
        ctx = _provider(transport=lambda u, p, t: _payload(uv_max=1.0)).fetch("X")
        assert ctx.uv_note == ""

    def test_high_uv_alone_makes_it_relevant(self):
        """Sun advice is worth giving even when the complaint is unrelated."""
        ctx = _provider(transport=lambda u, p, t: _payload(
            uv_max=9.0, dewpoint=12.0, heatindex=25.0, mintemp=20.0, epa=1)).fetch("X")
        assert ctx.is_clinically_relevant is True


# ---------------------------------------------------------------------------
# Moisture — dew point, not relative humidity
# ---------------------------------------------------------------------------

class TestMoisture:
    def test_very_dry_air_flagged_for_barrier_loss(self):
        """Leh: dew point 0.9 C — xerosis and eczema territory."""
        ctx = _provider(transport=lambda u, p, t: _payload(dewpoint=0.9)).fetch("Leh")
        assert "very dry" in ctx.moisture_note
        assert "barrier" in ctx.moisture_note
        assert ctx.is_clinically_relevant is True

    def test_occlusive_air_flagged_for_damp_folds(self):
        """Chennai: dew point 24.4 C — sweat cannot evaporate."""
        ctx = _provider(transport=lambda u, p, t: _payload(dewpoint=24.4)).fetch("Chennai")
        assert "sweat evaporates poorly" in ctx.moisture_note
        assert ctx.is_clinically_relevant is True

    def test_comfortable_dew_point_carries_no_note(self):
        ctx = _provider(transport=lambda u, p, t: _payload(dewpoint=12.0)).fetch("X")
        assert ctx.moisture_note == ""

    def test_dew_point_beats_relative_humidity(self):
        """
        Kochi 86% RH and Bikaner 40% RH look opposite, but dew points 23.0 and
        21.0 are close — both mean sweat will not evaporate. RH without
        temperature misleads, so the gate must key on dew point.
        """
        kochi = _provider(transport=lambda u, p, t: _payload(dewpoint=23.0, humidity=86)).fetch("K")
        bikaner = _provider(transport=lambda u, p, t: _payload(dewpoint=21.0, humidity=40)).fetch("B")
        assert kochi.moisture_note == bikaner.moisture_note != ""


# ---------------------------------------------------------------------------
# Thermal
# ---------------------------------------------------------------------------

class TestThermal:
    def test_heat_stress_flagged(self):
        """Bikaner: heat index 41.7 C — miliaria, hyperhidrosis."""
        ctx = _provider(transport=lambda u, p, t: _payload(heatindex=41.7)).fetch("Bikaner")
        assert "Heat stress high" in ctx.thermal_note
        assert ctx.is_clinically_relevant is True

    def test_cold_night_flagged_for_chilblains(self):
        ctx = _provider(transport=lambda u, p, t: _payload(
            uv_max=2.0, dewpoint=8.0, heatindex=12.0, mintemp=6.0, epa=1)).fetch("X")
        assert "Cold overnight" in ctx.thermal_note
        assert ctx.is_clinically_relevant is True

    def test_temperate_day_carries_no_thermal_note(self):
        ctx = _provider(transport=lambda u, p, t: _payload(
            heatindex=26.0, mintemp=19.0)).fetch("X")
        assert ctx.thermal_note == ""


# ---------------------------------------------------------------------------
# Gating — must stay quiet when nothing is noteworthy
# ---------------------------------------------------------------------------

class TestGating:
    def test_benign_day_is_not_injected(self):
        """Mild, clean, moderate UV — nothing to say, so say nothing."""
        ctx = _provider(transport=lambda u, p, t: _payload(
            uv_max=4.0, dewpoint=13.0, heatindex=26.0, mintemp=20.0,
            humidity=55, epa=1)).fetch("X")
        assert ctx.is_clinically_relevant is False

    def test_particulates_alone_make_it_relevant(self):
        ctx = _provider(transport=lambda u, p, t: _payload(
            uv_max=3.0, dewpoint=13.0, heatindex=26.0, mintemp=20.0, epa=4)).fetch("X")
        assert ctx.is_clinically_relevant is True


# ---------------------------------------------------------------------------
# Fail-open — every failure path individually
# ---------------------------------------------------------------------------

class TestFailsOpen:
    def test_no_api_key_returns_none_and_never_calls_out(self):
        calls = []
        p = EnvironmentalProvider(api_key=None, transport=lambda u, q, t: calls.append(1))
        assert p.fetch("Delhi") is None
        assert p.enabled is False and calls == []

    def test_blank_location_returns_none(self):
        assert _provider().fetch("") is None

    def test_timeout_returns_none(self):
        def boom(u, q, t):
            raise httpx.TimeoutException("timed out")
        assert _provider(transport=boom).fetch("Delhi") is None

    def test_http_error_returns_none(self):
        def boom(u, q, t):
            raise httpx.HTTPStatusError(
                "400", request=httpx.Request("GET", "http://x"),
                response=httpx.Response(400))
        assert _provider(transport=boom).fetch("110001") is None

    def test_unexpected_exception_returns_none(self):
        def boom(u, q, t):
            raise RuntimeError("odd")
        assert _provider(transport=boom).fetch("Delhi") is None

    @pytest.mark.parametrize("bad", [
        {}, {"location": {}}, {"current": {}}, None, [], "not a dict",
        {"location": {"name": "X"}}, {"location": None, "current": None},
    ])
    def test_malformed_payload_returns_none(self, bad):
        assert _provider(transport=lambda u, q, t: bad).fetch("Delhi") is None

    def test_missing_forecast_still_yields_current_conditions(self):
        """Losing the daily aggregate must not cost the whole reading."""
        pl = _payload(); pl.pop("forecast")
        ctx = _provider(transport=lambda u, q, t: pl).fetch("Delhi")
        assert ctx is not None
        assert ctx.dewpoint_c == 24.0

    def test_failure_is_not_cached(self):
        state = {"fail": True}
        def flaky(u, q, t):
            if state["fail"]:
                raise httpx.TimeoutException("x")
            return _payload()
        p = _provider(transport=flaky)
        assert p.fetch("Delhi") is None
        state["fail"] = False
        assert p.fetch("Delhi") is not None


# ---------------------------------------------------------------------------
# Location normalisation
# ---------------------------------------------------------------------------

class TestLocationNormalisation:
    def test_bare_city_gets_default_country(self):
        """`q=Delhi` alone resolves to Delhi, Ontario, Canada."""
        assert normalise_location("Delhi") == "Delhi, India"

    def test_already_qualified_is_left_alone(self):
        assert normalise_location("Kochi, India") == "Kochi, India"

    @pytest.mark.parametrize("coords", ["28.6139,77.2090", " 34.15 , 77.57 "])
    def test_coordinates_pass_through(self, coords):
        out = normalise_location(coords)
        assert "India" not in out and " " not in out

    def test_blank_is_blank(self):
        assert normalise_location("  ") == ""


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_second_call_served_from_cache(self):
        calls = []
        p = _provider(transport=lambda u, q, t: (calls.append(1), _payload())[1])
        a, b = p.fetch("Delhi"), p.fetch("Delhi")
        assert a is b and len(calls) == 1

    def test_distinct_locations_do_not_collide(self):
        calls = []
        p = _provider(transport=lambda u, q, t: (calls.append(1), _payload())[1])
        p.fetch("Leh"); p.fetch("Kochi")
        assert len(calls) == 2

    def test_expiry_refetches(self):
        calls = []
        p = _provider(transport=lambda u, q, t: (calls.append(1), _payload())[1],
                      cache_ttl_s=0.01)
        p.fetch("Delhi")
        import time as _t; _t.sleep(0.05)
        p.fetch("Delhi")
        assert len(calls) == 2

    def test_cache_is_bounded(self):
        p = _provider()
        for i in range(60):
            p.fetch(f"City{i}, India")
        assert p.cache_size <= 512


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

class TestPromptBlock:
    def test_uv_leads_the_block(self):
        """UV is the headline for a skin service, so it comes first."""
        block = _provider().fetch("Delhi").to_prompt_block()
        lines = block.split("\n")
        assert lines[0] == "[LOCAL CONDITIONS]"
        assert "UV index today (max)" in lines[2]

    def test_block_stays_compact(self):
        block = _provider().fetch("Delhi").to_prompt_block()
        assert len(block) < 800, "must not eat the prompt budget"

    def test_block_states_facts_without_asserting_causation(self):
        block = _provider().fetch("Delhi").to_prompt_block().lower()
        for claim in ("caused by", "due to the", "diagnos", "you have"):
            assert claim not in block

    def test_block_defers_weighing_to_the_exposure_layer(self):
        block = _provider().fetch("Delhi").to_prompt_block()
        # Reworded deliberately: the old blanket "never assume exposure" was
        # wrong for ambient conditions nobody can avoid. See
        # TestAmbientVersusBehaviouralFraming.
        assert "depend on behaviour and protection, so ask" in block
        assert "EXPOSURE HISTORY" in block

    def test_clean_air_omits_the_particulate_line(self):
        ctx = _provider(transport=lambda u, q, t: _payload(epa=1)).fetch("X")
        assert "Particulate load" not in ctx.to_prompt_block()

    def test_to_dict_round_trips_key_fields(self):
        d = _provider(transport=lambda u, q, t: _payload(uv_max=14.1)).fetch("Leh").to_dict()
        assert d["uv_index_max"] == 14.1
        assert d["uv_band"] == "Extreme"
        assert d["clinically_relevant"] is True


# ---------------------------------------------------------------------------
# Lookback — skin lesions lag exposure by 1-3 days
# ---------------------------------------------------------------------------

class TestLookback:
    """
    A patient presenting today with a burnt face was burnt YESTERDAY. Today's
    UV describes the exposure that has not yet produced symptoms.
    """

    def _mild_today(self):
        """Benign today: nothing in the current reading would fire the gate."""
        return _payload(uv_max=3.0, dewpoint=13.0, heatindex=26.0,
                        mintemp=20.0, humidity=55, epa=1)

    def test_mild_today_after_a_scorching_week_is_still_relevant(self):
        """THE case the lookback exists for — would have been missed before."""
        p = EnvironmentalProvider(
            api_key="k", lookback_days=7,
            transport=_routing(self._mild_today(),
                               _history([15.6, 12.0, 9.1, 5.3, 7.4, 8.2, 11.0])))
        ctx = p.fetch("Leh")
        assert ctx.uv_index_max == 3.0, "today really is mild"
        assert ctx.recent.peak_uv == 15.6
        # >=8.0 on five of the seven days: 15.6, 12.0, 9.1, 8.2, 11.0
        assert ctx.recent.high_uv_days == 5
        assert ctx.is_clinically_relevant is True

    def test_peak_uv_appears_in_the_block_with_the_lag_explained(self):
        p = EnvironmentalProvider(
            api_key="k", lookback_days=7,
            transport=_routing(self._mild_today(), _history([15.6, 3.0, 3.0])))
        block = p.fetch("Leh").to_prompt_block()
        assert "peak UV 16" in block
        assert "lag exposure by 1-3 days" in block

    def test_dry_spell_detected_from_the_window(self):
        p = EnvironmentalProvider(
            api_key="k", lookback_days=7,
            transport=_routing(self._mild_today(),
                               _history([3.0] * 5, humidities=[70, 60, 33, 45, 50])))
        ctx = p.fetch("X")
        assert ctx.recent.min_avg_humidity_pct == 33
        assert ctx.recent.had_dry_spell is True
        assert ctx.is_clinically_relevant is True
        assert "humidity down to 33%" in ctx.to_prompt_block()

    def test_calm_week_after_calm_day_stays_irrelevant(self):
        """The control: a genuinely unremarkable week must not fire."""
        p = EnvironmentalProvider(
            api_key="k", lookback_days=7,
            transport=_routing(self._mild_today(),
                               _history([2.0, 3.0, 4.0], humidities=[60, 65, 70])))
        ctx = p.fetch("X")
        assert ctx.recent.had_high_uv is False
        assert ctx.recent.had_dry_spell is False
        assert ctx.is_clinically_relevant is False
        assert ctx.recent.summary() == ""

    def test_history_failure_does_not_lose_today(self):
        """Losing the window must never cost the current reading."""
        p = EnvironmentalProvider(
            api_key="k", lookback_days=7,
            transport=_routing(_payload(uv_max=9.0), history=None))
        ctx = p.fetch("X")
        assert ctx is not None
        assert ctx.recent is None
        assert ctx.uv_index_max == 9.0

    def test_lookback_uses_one_range_call_not_n(self):
        urls = []
        def _t(url, params, timeout):
            urls.append(url)
            return _history([9.0] * 7) if "history" in url else _payload()
        EnvironmentalProvider(api_key="k", lookback_days=7, transport=_t).fetch("X")
        assert sum("history" in u for u in urls) == 1, "must be a single range query"

    def test_lookback_disabled_makes_no_history_call(self):
        urls = []
        def _t(url, params, timeout):
            urls.append(url)
            return _payload()
        EnvironmentalProvider(api_key="k", lookback_days=0, transport=_t).fetch("X")
        assert not any("history" in u for u in urls)

    @pytest.mark.parametrize("bad", [
        None, {}, {"forecast": None}, {"forecast": {"forecastday": []}}, "junk",
        {"forecast": {"forecastday": [{"day": None}]}},
        {"forecast": {"forecastday": [{"day": {"avghumidity": 40}}]}},   # no uv
    ])
    def test_malformed_history_degrades_to_none(self, bad):
        p = EnvironmentalProvider(
            api_key="k", lookback_days=7, transport=_routing(_payload(), bad))
        ctx = p.fetch("X")
        assert ctx is not None and ctx.recent is None

    def test_partial_day_records_do_not_crash(self):
        """One malformed day among good ones must not lose the window."""
        hist = _history([9.0, 12.0])
        hist["forecast"]["forecastday"].append({"day": {"uv": "bad", "avghumidity": None}})
        p = EnvironmentalProvider(
            api_key="k", lookback_days=7, transport=_routing(_payload(), hist))
        ctx = p.fetch("X")
        assert ctx.recent.peak_uv == 12.0

    def test_to_dict_exposes_the_window(self):
        p = EnvironmentalProvider(
            api_key="k", lookback_days=7,
            transport=_routing(_payload(), _history([15.6, 3.0])))
        d = p.fetch("X").to_dict()
        assert d["recent"]["peak_uv"] == 15.6
        assert d["recent"]["had_high_uv"] is True


# ---------------------------------------------------------------------------
# Country guard — the gazetteer will happily answer with the wrong continent
# ---------------------------------------------------------------------------

class TestCountryGuard:
    """
    WeatherAPI fuzzy-matches. "zzqqxx nowhere" (with ", India" appended by us)
    resolved to Nowhere, OKLAHOMA, USA and returned a full reading — the
    service would have handed an Indian patient sun advice for the American
    Midwest. When WE supplied the country, the answer must honour it.
    """

    @staticmethod
    def _at(country, name="Nowhere", region="Oklahoma"):
        pl = _payload()
        pl["location"].update(name=name, region=region, country=country)
        return pl

    def test_unqualified_input_resolving_abroad_is_discarded(self):
        p = _provider(transport=lambda u, q, t:
                      self._at("United States of America"))
        assert p.fetch("zzqqxx nowhere") is None

    def test_unqualified_input_resolving_in_country_is_kept(self):
        p = _provider(transport=lambda u, q, t:
                      self._at("India", name="Delhi", region="Delhi"))
        ctx = p.fetch("Delhi")
        assert ctx is not None and ctx.location_label.endswith("India")

    def test_explicit_foreign_country_is_honoured(self):
        """A caller who names a country means it — do not second-guess them."""
        p = _provider(transport=lambda u, q, t:
                      self._at("United Kingdom", name="London", region="Greater London"))
        ctx = p.fetch("London, UK")
        assert ctx is not None, "explicit foreign location must not be discarded"

    def test_coordinates_are_never_country_checked(self):
        """lat/lon is unambiguous by construction."""
        p = _provider(transport=lambda u, q, t:
                      self._at("United States of America"))
        assert p.fetch("35.15,-98.44") is not None

    def test_guard_respects_a_configured_default_country(self):
        p = _provider(transport=lambda u, q, t:
                      self._at("United Kingdom", name="Manchester", region="England"),
                      default_country="UK")
        assert p.fetch("Manchester") is not None

    def test_mismatch_is_logged_not_silent(self, caplog):
        p = _provider(transport=lambda u, q, t:
                      self._at("United States of America"))
        with caplog.at_level("WARNING"):
            p.fetch("nowhere")
        assert "outside the expected country" in caplog.text


# ---------------------------------------------------------------------------
# Ambient fact vs behavioural exposure
# ---------------------------------------------------------------------------

class TestAmbientVersusBehaviouralFraming:
    """
    Not every environmental parameter is an "exposure".

    Ambient temperature and humidity are unavoidable for anyone living in the
    place — indoors too, since heating air lowers its relative humidity. So
    "have you been exposed to dry air?" is an unanswerable question, and the
    model asked exactly that during a Leh consultation while holding a reading
    of dew point 4C. UV is the opposite: it needs the patient outdoors,
    unprotected, at the right hours, so it must never be assumed.

    The block therefore has to carry BOTH instructions at once.
    """

    def _block(self):
        return _provider(transport=lambda u, q, t: _payload()).fetch("Leh").to_prompt_block()

    def test_ambient_conditions_are_stated_not_asked_about(self):
        b = self._block().lower()
        assert "unavoidable here, indoors too" in b
        assert "rather than asking whether the air is dry or cold" in b

    def test_behavioural_modifiers_are_what_gets_asked(self):
        b = self._block().lower()
        # The modifier checklist lives in the EXPOSURE HISTORY layer, not here
        # — duplicating it in every block would spend budget for nothing.
        assert "see exposure history" in b

    def test_uv_and_particulates_still_require_asking(self):
        b = self._block().lower()
        assert "uv and particulate exposure depend on behaviour" in b

    def test_block_still_refuses_to_conclude(self):
        assert "not a conclusion" in self._block().lower()
