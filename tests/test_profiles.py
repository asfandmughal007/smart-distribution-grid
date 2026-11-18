"""Tests for the synthetic load, PV and price profiles."""

import numpy as np
import pandas as pd
import pytest

from src.profiles.load_profile import load_profile_from_config
from src.profiles.price_profile import price_profile_from_config
from src.profiles.pv_profile import clear_sky_shape, generate_pv_profile, pv_profile_from_config
from src.utils.config import time_index


@pytest.fixture(scope="module")
def timestamps(config):
    return time_index(config)


def test_time_index_has_96_steps(timestamps):
    assert len(timestamps) == 96
    assert (np.diff(timestamps.asi8) == pd.Timedelta(minutes=15).value).all()


def test_load_profile_positive_and_normalized(config, timestamps):
    load = load_profile_from_config(config, timestamps, commercial_share=0.4)
    assert len(load) == 96
    assert (load > 0).all().all()
    assert load.residential_multiplier.max() == pytest.approx(1.0)
    assert load.commercial_multiplier.max() == pytest.approx(1.0)


def test_load_profile_shape(config, timestamps):
    """Residential demand peaks in the evening, commercial demand during office hours."""
    load = load_profile_from_config(config, timestamps, commercial_share=0.4)
    assert 17 <= load.residential_multiplier.idxmax().hour <= 22
    assert 8 <= load.commercial_multiplier.idxmax().hour <= 16
    night = load.between_time("02:00", "04:00")
    assert night.residential_multiplier.max() < 0.5


def test_pv_zero_at_night(config, timestamps):
    pv = pv_profile_from_config(config, timestamps, installed_capacity_kw=1000)
    hours = pv.index.hour + pv.index.minute / 60
    night = (hours < config["pv"]["sunrise_h"]) | (hours >= config["pv"]["sunset_h"])
    assert (pv.loc[night, "pv_kw"] == 0).all()
    assert len(pv) == 96
    assert pv.pv_kw.max() <= 1000 * config["pv"]["performance_ratio"] + 1e-9
    assert 11 <= pv.pv_kw.idxmax().hour <= 15


def test_pv_deterministic(timestamps):
    a = generate_pv_profile(timestamps, 500, seed=7)
    b = generate_pv_profile(timestamps, 500, seed=7)
    c = generate_pv_profile(timestamps, 500, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.pv_kw, c.pv_kw)


def test_clear_sky_rejects_invalid_daylight(timestamps):
    with pytest.raises(ValueError):
        clear_sky_shape(timestamps, sunrise_h=20, sunset_h=6)


def test_price_profile(config, timestamps):
    price = price_profile_from_config(config, timestamps)
    assert len(price) == 96
    assert (price.import_eur_per_kwh > price.export_eur_per_kwh).all()
    evening = price.between_time("18:00", "20:45").import_eur_per_kwh.mean()
    night = price.between_time("01:00", "04:45").import_eur_per_kwh.mean()
    assert evening > night
