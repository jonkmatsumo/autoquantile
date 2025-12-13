from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.utils.geo_utils import GeoMapper


@pytest.fixture
def mock_config():
    return {
        "mappings": {"location_targets": {"New York, NY": 1, "San Francisco, CA": 2}},
        "location_settings": {"max_distance_km": 50},
    }


def test_geo_mapper_init(mock_config):
    with (
        patch("src.utils.geo_utils.Nominatim"),
        patch("builtins.open", mock_open(read_data="{}")),
        patch("src.utils.geo_utils.GeoMapper._init_targets"),
    ):
        mapper = GeoMapper(config=mock_config)
        assert mapper.targets == mock_config["mappings"]["location_targets"]
        assert mapper.settings["max_distance_km"] == 50


def test_get_zone_cached(mock_config):
    with (
        patch("src.utils.geo_utils.Nominatim"),
        patch("builtins.open", mock_open(read_data="{}")),
        patch("src.utils.geo_utils.GeoMapper._init_targets"),
    ):
        mapper = GeoMapper(config=mock_config)
        mapper.zone_cache["Test City"] = 3
        assert mapper.get_zone("Test City") == 3


def test_get_zone_proximity(mock_config):
    with (
        patch("src.utils.geo_utils.Nominatim") as MockNominatim,
        patch("builtins.open", mock_open(read_data="{}")),
        patch("src.utils.geo_utils.GeoMapper._init_targets"),
    ):
        mapper = GeoMapper(config=mock_config)
        mapper.target_coords = {"New York, NY": (40.7128, -74.0060)}
        mock_geolocator = MockNominatim.return_value
        mock_location = MagicMock()
        mock_location.latitude = 40.7357
        mock_location.longitude = -74.1724
        mock_geolocator.geocode.return_value = mock_location
        zone = mapper.get_zone("Newark, NJ")
        assert zone == 1
        assert mapper.zone_cache["Newark, NJ"] == 1


def test_get_zone_far(mock_config):
    with (
        patch("src.utils.geo_utils.Nominatim") as MockNominatim,
        patch("builtins.open", mock_open(read_data="{}")),
        patch("src.utils.geo_utils.GeoMapper._init_targets"),
    ):
        mapper = GeoMapper(config=mock_config)
        mapper.target_coords = {"New York, NY": (40.7128, -74.0060)}
        mock_geolocator = MockNominatim.return_value
        mock_location = MagicMock()
        mock_location.latitude = 51.5074
        mock_location.longitude = -0.1278
        mock_geolocator.geocode.return_value = mock_location
        zone = mapper.get_zone("London, UK")
        assert zone == 4
