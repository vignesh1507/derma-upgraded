"""Environmental context (UV, moisture, air quality) for a dermatology service."""

from app.services.environment.models import (
    CHILBLAIN_MIN_TEMP_C,
    HEAT_INDEX_SWEAT_C,
    OCCLUSIVE_DEWPOINT_C,
    PM_RELEVANT_EPA_BAND,
    UV_PHOTOSENSITIVE_INDEX,
    UV_RELEVANT_INDEX,
    VERY_DRY_DEWPOINT_C,
    EnvironmentalContext,
)
from app.services.environment.provider import (
    EnvironmentalProvider,
    normalise_location,
)

__all__ = [
    "EnvironmentalContext",
    "EnvironmentalProvider",
    "normalise_location",
    "UV_RELEVANT_INDEX",
    "UV_PHOTOSENSITIVE_INDEX",
    "VERY_DRY_DEWPOINT_C",
    "OCCLUSIVE_DEWPOINT_C",
    "HEAT_INDEX_SWEAT_C",
    "CHILBLAIN_MIN_TEMP_C",
    "PM_RELEVANT_EPA_BAND",
]
