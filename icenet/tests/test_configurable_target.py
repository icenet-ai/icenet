#!/usr/bin/env python
"""Tests for the configurable forecast target.

These cover the two properties that the target implementation must hold:

1. The target is selected from its own dataset by timestamp, so it is never
   derived from a positional index into the predictor time axis.
2. The target window starts at the step *after* the initialisation date, so an
   initialisation date can never be handed to the network as its own label.
"""

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from icenet.data.loaders.base import (DEFAULT_TARGET_VARIABLE,
                                      IceNetBaseDataLoader,
                                      _resolve_target_config)
from icenet.data.loaders.dask import generate_sample
from icenet.data.loaders.utils import IceNetDataWarning

SHAPE = (2, 2)


def _make_dataset(variable: str, values: object, dates: object) -> xr.Dataset:
    """Build a minimal (yc, xc, time) dataset with one constant field per date."""
    data = np.stack([
        np.full(SHAPE, value, dtype=np.float32) for value in values
    ],
                    axis=-1)

    return xr.Dataset(
        {variable: (("yc", "xc", "time"), data)},
        coords={
            "yc": [0, 1],
            "xc": [0, 1],
            "time": pd.to_datetime(dates),
        },
    )


def _write_mask(tmp_path: object, name: str, array: object) -> str:
    """Persist a mask DataArray and return its path, as masks are read by path."""
    path = str(tmp_path / "{}.nc".format(name))
    array.rename(name).to_netcdf(path, engine="h5netcdf")
    return path


def _call(forecast_date: object,
          var_ds: object,
          target_ds: object,
          *,
          channels: object = None,
          n_forecast_steps: int = 2,
          num_channels: int = 1,
          masks: object = None,
          target: object = None,
          missing_dates: object = None,
          prediction: bool = False) -> tuple:
    """Invoke generate_sample with the defaults these tests share."""
    x, y, sample_weights = generate_sample(
        forecast_date,
        var_ds,
        {},
        None,
        target_ds,
        channels if channels is not None else {"predictor_abs": 1},
        np.float32,
        False,
        [],
        missing_dates if missing_dates is not None else [],
        n_forecast_steps,
        num_channels,
        SHAPE,
        {},
        "day",
        masks if masks is not None else {},
        target if target is not None else {
            "source": "era5",
            "variable": "tas_abs",
            "mask": None,
            "weight_mask": None,
        },
        prediction,
    )

    return dask.compute(x, y, sample_weights)


# Target resolution.


def _config(sources: object = None, **kwargs) -> dict:
    config = {
        "sources": sources if sources is not None else {
            "osisaf": {
                "processed_files": {
                    DEFAULT_TARGET_VARIABLE: ["siconca_abs.nc"],
                }
            }
        }
    }
    config.update(kwargs)
    return config


def test_legacy_configuration_infers_siconca_target():
    """A configuration with no target block keeps the historical target."""
    target = _resolve_target_config(_config())

    assert target["source"] == "osisaf"
    assert target["variable"] == DEFAULT_TARGET_VARIABLE
    # No masks are declared, so no masking is applied.
    assert target["mask"] is None
    assert target["weight_mask"] is None


def test_legacy_configuration_retains_legacy_masks():
    """Where the legacy masks exist, legacy masking behaviour is preserved."""
    target = _resolve_target_config(
        _config(masks={
            "land": {},
            "active_grid_cell": {}
        }))

    assert target["mask"] == "land"
    assert target["weight_mask"] == "active_grid_cell"


def test_ambiguous_legacy_target_is_rejected():
    """Guessing between sources would silently change the ground truth."""
    sources = {
        "osisaf": {
            "processed_files": {
                DEFAULT_TARGET_VARIABLE: ["a.nc"]
            }
        },
        "amsr": {
            "processed_files": {
                DEFAULT_TARGET_VARIABLE: ["b.nc"]
            }
        },
    }

    with pytest.raises(RuntimeError, match="exactly one"):
        _resolve_target_config(_config(sources=sources))


def test_missing_legacy_target_is_rejected():
    sources = {"era5": {"processed_files": {"tas_abs": ["tas.nc"]}}}

    with pytest.raises(RuntimeError, match="exactly one"):
        _resolve_target_config(_config(sources=sources))


def test_explicit_target_resolves_source_variable_and_mask():
    config = _config(sources={
        "era5": {
            "processed_files": {
                "tas_abs": ["tas1.nc", "tas2.nc"]
            }
        }
    },
                     masks={
                         "land": {},
                         "active_grid_cell": {}
                     })

    target = _resolve_target_config(config,
                                    override={
                                        "source": "era5",
                                        "variable": "tas_abs",
                                        "mask": "land",
                                    })

    assert target["source"] == "era5"
    assert target["variable"] == "tas_abs"
    assert target["mask"] == "land"
    # Not defaulted to the legacy mask: sea ice weighting is meaningless over
    # an arbitrary field, so an explicit target must ask for it.
    assert target["weight_mask"] is None


def test_unknown_target_source_is_rejected():
    with pytest.raises(ValueError, match="Unknown target source"):
        _resolve_target_config(_config(),
                               override={
                                   "source": "nope",
                                   "variable": DEFAULT_TARGET_VARIABLE
                               })


def test_unknown_target_variable_is_rejected():
    with pytest.raises(ValueError, match="not available in"):
        _resolve_target_config(_config(),
                               override={
                                   "source": "osisaf",
                                   "variable": "tas_abs"
                               })


def test_incomplete_target_is_rejected():
    with pytest.raises(ValueError, match="missing required fields"):
        _resolve_target_config(_config(), override={"source": "osisaf"})


def test_non_mapping_target_is_rejected():
    with pytest.raises(TypeError, match="JSON object"):
        _resolve_target_config(_config(), override="siconca_abs")


def test_unknown_target_mask_is_rejected():
    with pytest.raises(ValueError, match="unknown mask"):
        _resolve_target_config(_config(masks={"land": {}}),
                               override={
                                   "source": "osisaf",
                                   "variable": DEFAULT_TARGET_VARIABLE,
                                   "mask": "not_a_mask",
                               })


# Temporal alignment.


def test_configured_target_is_selected_by_forecast_date():
    """The target opens on t, the lags close before it, and it is not an input."""
    dates = pd.date_range("2020-01-01", periods=4, freq="D")

    var_ds = _make_dataset("predictor_abs", [1.0, 2.0, 3.0, 4.0], dates)
    target_ds = _make_dataset("tas_abs", [10.0, 20.0, 30.0, 40.0], dates)

    assert "tas_abs" not in var_ds.data_vars

    x, y, weights = _call(pd.Timestamp("2020-01-02"), var_ds, target_ds)

    # The most recent lag is t-1. Reading 2.0 here would mean the lag window
    # had been opened on the initialisation date itself.
    np.testing.assert_allclose(x[:, :, 0], 1.0)

    np.testing.assert_allclose(y[:, :, 0, 0], 20.0)
    np.testing.assert_allclose(y[:, :, 1, 0], 30.0)

    np.testing.assert_allclose(weights, 1.0)


def test_lag_window_never_overlaps_the_target_window():
    """The #256 invariant: no predictor may be drawn from a target step."""
    dates = pd.date_range("2020-01-01", periods=6, freq="D")

    # One variable acting as both predictor and target is the case that lets
    # the network trivially copy its input across to its output.
    ds = _make_dataset("siconca_abs", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dates)

    x, y, _ = _call(pd.Timestamp("2020-01-04"),
                    ds,
                    ds,
                    channels={"siconca_abs": 2},
                    num_channels=2,
                    target={
                        "source": "osisaf",
                        "variable": "siconca_abs",
                        "mask": None,
                        "weight_mask": None,
                    })

    lags = set(np.unique(x).tolist())
    targets = set(np.unique(y).tolist())

    assert lags == {2.0, 3.0}, lags
    assert targets == {4.0, 5.0}, targets
    assert not lags & targets


def test_target_axis_need_not_align_with_predictor_axis():
    """A target file holding none of the input history still resolves by date."""
    var_dates = pd.date_range("2020-01-01", periods=4, freq="D")
    target_dates = pd.date_range("2020-01-02", periods=4, freq="D")

    var_ds = _make_dataset("predictor_abs", [1.0, 2.0, 3.0, 4.0], var_dates)
    target_ds = _make_dataset("tas_abs", [100.0, 200.0, 300.0, 400.0],
                              target_dates)

    _, y, _ = _call(pd.Timestamp("2020-01-02"), var_ds, target_ds)

    # These dates are positions 0 and 1 of the target axis but 1 and 2 of the
    # predictor axis, which would instead yield 200.0 and 300.0.
    np.testing.assert_allclose(y[:, :, 0, 0], 100.0)
    np.testing.assert_allclose(y[:, :, 1, 0], 200.0)


def test_missing_target_date_raises_rather_than_misaligning():
    """A truncated target must drop the sample, not shift the target window."""
    var_dates = pd.date_range("2020-01-01", periods=4, freq="D")
    target_dates = pd.date_range("2020-01-01", periods=2, freq="D")

    var_ds = _make_dataset("predictor_abs", [1.0, 2.0, 3.0, 4.0], var_dates)
    target_ds = _make_dataset("tas_abs", [10.0, 20.0], target_dates)

    with pytest.raises(IceNetDataWarning, match="Unable to construct target"):
        _call(pd.Timestamp("2020-01-02"), var_ds, target_ds)


def test_operational_prediction_runs_past_the_end_of_the_data():
    """An initialisation date beyond the observations still yields lags."""
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    var_ds = _make_dataset("predictor_abs", [1.0, 2.0, 3.0, 4.0], dates)

    x, _, _ = _call(pd.Timestamp("2020-01-05"),
                    var_ds,
                    None,
                    prediction=True)

    # Forecasting the first unobserved step, so the last observation is t-1.
    np.testing.assert_allclose(x[:, :, 0], 4.0)


def test_prediction_requires_no_target():
    """Prediction samples have no ground truth and must not need a target file."""
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    var_ds = _make_dataset("predictor_abs", [1.0, 2.0, 3.0, 4.0], dates)

    x, y, weights = _call(pd.Timestamp("2020-01-02"),
                          var_ds,
                          None,
                          prediction=True)

    np.testing.assert_allclose(x[:, :, 0], 1.0)
    np.testing.assert_allclose(y, 0.0)
    np.testing.assert_allclose(weights, 1.0)


# Masking.


def test_static_mask_zeroes_output_and_weights(tmp_path):
    dates = pd.date_range("2020-01-01", periods=4, freq="D")

    var_ds = _make_dataset("predictor_abs", [1.0, 2.0, 3.0, 4.0], dates)
    target_ds = _make_dataset("tas_abs", [10.0, 20.0, 30.0, 40.0], dates)

    land = xr.DataArray(np.array([[False, True], [False, False]]),
                        dims=("yc", "xc"))
    masks = {"land": _write_mask(tmp_path, "land", land)}

    _, y, weights = _call(pd.Timestamp("2020-01-02"),
                          var_ds,
                          target_ds,
                          masks=masks,
                          target={
                              "source": "era5",
                              "variable": "tas_abs",
                              "mask": "land",
                              "weight_mask": None,
                          })

    # Masked cell is excluded from both the label and the loss.
    np.testing.assert_allclose(y[0, 1, :, 0], 0.0)
    np.testing.assert_allclose(weights[0, 1, :, 0], 0.0)

    # Unmasked cells are untouched.
    np.testing.assert_allclose(y[0, 0, 0, 0], 20.0)
    np.testing.assert_allclose(weights[0, 0, :, 0], 1.0)


def test_weight_mask_follows_the_target_step_not_the_initialisation(tmp_path):
    """A month-varying weight mask is selected per target step."""
    dates = pd.to_datetime(
        ["2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02"])

    var_ds = _make_dataset("predictor_abs", [1.0, 2.0, 3.0, 4.0], dates)
    target_ds = _make_dataset("tas_abs", [10.0, 20.0, 30.0, 40.0], dates)

    # Each month weighs its own number, so the selected month is readable
    # straight off the result.
    weight = xr.DataArray(np.stack([
        np.full(SHAPE, float(month), dtype=np.float32)
        for month in range(1, 13)
    ]),
                          dims=("month", "yc", "xc"),
                          coords={"month": list(range(1, 13))})
    masks = {
        "active_grid_cell": _write_mask(tmp_path, "active_grid_cell", weight)
    }

    _, _, weights = _call(pd.Timestamp("2020-01-31"),
                          var_ds,
                          target_ds,
                          masks=masks,
                          target={
                              "source": "era5",
                              "variable": "tas_abs",
                              "mask": None,
                              "weight_mask": "active_grid_cell",
                          })

    # The target steps straddle a month boundary, so each leadtime picks up its
    # own month rather than sharing the initialisation month.
    np.testing.assert_allclose(weights[:, :, 0, 0], 1.0)
    np.testing.assert_allclose(weights[:, :, 1, 0], 2.0)


def test_mask_of_the_wrong_shape_is_not_swallowed(tmp_path):
    """generate_and_write drops IceNetDataWarning, so a bad mask must not be one.

    Otherwise a misconfigured mask yields an empty dataset and a log line.
    """
    dates = pd.date_range("2020-01-01", periods=4, freq="D")

    var_ds = _make_dataset("predictor_abs", [1.0, 2.0, 3.0, 4.0], dates)
    target_ds = _make_dataset("tas_abs", [10.0, 20.0, 30.0, 40.0], dates)

    land = xr.DataArray(np.zeros((3, 3), dtype=bool), dims=("yc", "xc"))
    masks = {"land": _write_mask(tmp_path, "land", land)}

    with pytest.raises(RuntimeError, match="expected"):
        _call(pd.Timestamp("2020-01-02"),
              var_ds,
              target_ds,
              masks=masks,
              target={
                  "source": "era5",
                  "variable": "tas_abs",
                  "mask": "land",
                  "weight_mask": None,
              })


def test_nan_target_is_excluded_from_the_loss():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")

    var_ds = _make_dataset("predictor_abs", [1.0, 2.0, 3.0, 4.0], dates)
    target_ds = _make_dataset("tas_abs", [10.0, 20.0, 30.0, 40.0], dates)
    target_ds["tas_abs"][0, 1, 1] = np.nan

    _, y, weights = _call(pd.Timestamp("2020-01-02"), var_ds, target_ds)

    np.testing.assert_allclose(weights[0, 1, 0, 0], 0.0)
    np.testing.assert_allclose(y[0, 1, 0, 0], 0.0)
    np.testing.assert_allclose(weights[0, 0, 0, 0], 1.0)


def test_missing_date_zeroes_the_weight():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")

    var_ds = _make_dataset("predictor_abs", [1.0, 2.0, 3.0, 4.0], dates)
    target_ds = _make_dataset("tas_abs", [10.0, 20.0, 30.0, 40.0], dates)

    _, _, weights = _call(pd.Timestamp("2020-01-02"),
                          var_ds,
                          target_ds,
                          missing_dates=[pd.Timestamp("2020-01-03")])

    # 2020-01-03 is the second target step, so only that leadtime drops out.
    np.testing.assert_allclose(weights[:, :, 0, 0], 1.0)
    np.testing.assert_allclose(weights[:, :, 1, 0], 0.0)


# Configuration round trip.


class _StubLoader:
    """The minimum surface _write_dataset_config touches."""

    identifier = "stub"
    north = True
    south = False
    num_channels = 1
    workers = 1

    def __init__(self, tmp_path: object, target: object):
        self._channels = {"predictor_abs": 1}
        self._configuration_path = str(tmp_path / "loader.stub.json")
        self._dataset_config_path = str(tmp_path)
        self._dtype = np.float32
        self._lag_time = 1
        self._lead_time = 2
        self._loss_weight_days = False
        self._missing_dates = []
        self._output_batch_size = 2
        self._path = str(tmp_path)
        self._shape = (2, 2)
        self._target = target
        self._var_lag_override = {}

    @property
    def target(self):
        return dict(self._target)


def test_generated_dataset_config_records_the_target(tmp_path):
    """Without this a regenerated loader cannot know what it predicted."""
    import orjson

    target = {
        "source": "era5",
        "variable": "tas_abs",
        "mask": None,
        "weight_mask": None,
    }
    stub = _StubLoader(tmp_path, target)

    IceNetBaseDataLoader._write_dataset_config(stub, {"train": 1})

    with open(tmp_path / "dataset_config.stub.json") as fh:
        written = orjson.loads(fh.read())

    assert written["target"] == target


def test_merged_dataset_rejects_conflicting_targets(monkeypatch):
    """Merging different targets would train a model with no single meaning."""
    from icenet.data import network_dataset as nd

    class _Loader:
        north = True
        south = False

    monkeypatch.setattr(nd.IceNetDataLoaderFactory, "create_data_loader",
                        lambda self, *args, **kwargs: _Loader())

    merged = nd.MergedIceNetDataSet.__new__(nd.MergedIceNetDataSet)
    merged._config = dict(loader_paths=[],
                          loaders=[],
                          north=False,
                          south=False)

    base = dict(channels=["predictor_abs_1"],
                counts={"train": 1},
                dataset_path="a",
                dtype="float32",
                identifier="a",
                lag_time=1,
                lead_time=2,
                loader_config="a.json",
                loss_weight_days=False,
                north=True,
                num_channels=1,
                output_batch_size=2,
                shape=[2, 2],
                south=False,
                target={
                    "source": "osisaf",
                    "variable": DEFAULT_TARGET_VARIABLE,
                    "mask": "land",
                    "weight_mask": "active_grid_cell",
                },
                var_lag_override={})

    merged._merge_configurations("a.json", base)

    conflicting = {
        **base, "target": {
            **base["target"], "variable": "tas_abs"
        }
    }

    with pytest.raises(RuntimeError, match="target is not the same"):
        merged._merge_configurations("b.json", conflicting)


def test_merged_dataset_accepts_matching_targets(monkeypatch):
    from icenet.data import network_dataset as nd

    class _Loader:
        north = True
        south = False

    monkeypatch.setattr(nd.IceNetDataLoaderFactory, "create_data_loader",
                        lambda self, *args, **kwargs: _Loader())

    merged = nd.MergedIceNetDataSet.__new__(nd.MergedIceNetDataSet)
    merged._config = dict(loader_paths=[],
                          loaders=[],
                          north=False,
                          south=False)

    base = dict(channels=["predictor_abs_1"],
                counts={"train": 1},
                dataset_path="a",
                dtype="float32",
                identifier="a",
                lag_time=1,
                lead_time=2,
                loader_config="a.json",
                loss_weight_days=False,
                north=True,
                num_channels=1,
                output_batch_size=2,
                shape=[2, 2],
                south=False,
                target={
                    "source": "era5",
                    "variable": "tas_abs",
                    "mask": None,
                    "weight_mask": None,
                },
                var_lag_override={})

    merged._merge_configurations("a.json", base)
    merged._merge_configurations("b.json", {**base, "identifier": "b"})

    assert merged._config["counts"]["train"] == 2
