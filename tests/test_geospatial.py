"""Geospatial tools: the arithmetic-on-coordinates capability."""

import math

from threat_detector.tools import geospatial


def test_haversine_known_distance():
    # Berlin -> Alexanderplatz is well under 1 km.
    berlin = (52.5200, 13.4050)
    alex = (52.5219, 13.4132)
    d = geospatial.haversine_km(berlin, alex)
    assert 0.5 < d < 1.5


def test_geocode_synthetic(synthetic_cfg):
    coords = geospatial.geocode("Berlin", synthetic_cfg)
    assert coords is not None
    assert math.isclose(coords[0], 52.52, abs_tol=0.01)


def test_proximity_close_site_is_high_severity(synthetic_cfg):
    findings = geospatial.assess_venue_proximity(
        "Berlin", ["Alexanderplatz, Berlin"], synthetic_cfg
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.detail["distance_km"] < 2.0
    assert f.severity > 0.5  # close => elevated


def test_same_name_city_resolves_to_different_countries(synthetic_cfg):
    # The ambiguity the geospatial agent exists to catch.
    assert geospatial.country_of("Saint Petersburg, Russia", synthetic_cfg) == "Russia"
    assert geospatial.country_of("Saint Petersburg, Florida", synthetic_cfg) == "United States"
