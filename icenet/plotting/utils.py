import datetime as dt
import glob
import logging
import os
import re

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

from functools import cache
from ibicus.debias import LinearScaling
from pyproj import CRS
from rasterio.enums import Resampling


def broadcast_forecast(start_date: object,
                       end_date: object,
                       datafiles: object = None,
                       dataset: object = None,
                       target: object = None) -> object:
    """

    :param start_date:
    :param end_date:
    :param datafiles:
    :param dataset:
    :param target:
    :return:
    """

    assert (datafiles is None) ^ (dataset is None), \
        "Only one of datafiles and dataset can be set"

    if datafiles:
        logging.info("Using {} to generate forecast through {} to {}".format(
            ", ".join(datafiles), start_date, end_date))
        dataset = xr.open_mfdataset(datafiles, engine="netcdf4")

    dates = pd.date_range(start_date, end_date)
    i = 0

    logging.debug("Dataset summary: \n{}".format(dataset))

    if len(dataset.time.values) > 1:
        while dataset.time.values[i + 1] < dates[0]:
            i += 1

    logging.info("Starting index will be {} for {} - {}".format(
        i, dates[0], dates[-1]))
    dt_arr = []

    for d in dates:
        logging.debug("Looking for date {}".format(d))
        arr = None

        while arr is None:
            if d >= dataset.time.values[i]:
                d_lead = (d - dataset.time.values[i]).days

                if i + 1 < len(dataset.time.values):
                    if pd.to_datetime(dataset.time.values[i]) + \
                            dt.timedelta(days=d_lead) >= \
                            pd.to_datetime(dataset.time.values[i + 1]) + \
                            dt.timedelta(days=1):
                        i += 1
                        continue

                logging.debug("Selecting date {} and lead {}".format(
                    pd.to_datetime(dataset.time.values[i]).strftime("%D"),
                    d_lead))

                arr = dataset.sel(time=dataset.time.values[i],
                                  leadtime=d_lead).\
                    copy().\
                    drop("time").\
                    assign_coords(dict(time=d)).\
                    drop("leadtime")
            else:
                i += 1

        dt_arr.append(arr)

    target_ds = xr.concat(dt_arr, dim="time")

    if target:
        logging.info("Saving dataset to {}".format(target))
        target_ds.to_netcdf(target)
    return target_ds


def get_seas_forecast_init_dates(
    hemisphere: str,
    source_path: object = os.path.join(".", "data", "mars.seas")
) -> object:
    """
    Obtains list of dates for which we have SEAS forecasts we have.

    :param hemisphere: string, typically either 'north' or 'south'
    :param source_path: path where north and south SEAS forecasts are stored

    :return: list of dates
    """
    # list the files in the path where SEAS forecasts are stored
    filenames = os.listdir(os.path.join(source_path, hemisphere, "siconca"))
    # obtain the dates from files with YYYYMMDD.nc format
    return pd.to_datetime(
        [x.split('.')[0] for x in filenames if re.search(r'^\d{8}\.nc$', x)])


def get_seas_forecast_da(
        hemisphere: str,
        date: str,
        bias_correct: bool = True,
        source_path: object = os.path.join(".", "data", "mars.seas"),
) -> tuple:
    """
    Atmospheric model Ensemble 15-day forecast (Set III - ENS)

Coordinates:
  * time                          (time) datetime64[ns] 2022-04-01 ... 2022-0...
  * yc                            (yc) float64 5.388e+06 ... -5.388e+06
  * xc                            (xc) float64 -5.388e+06 ... 5.388e+06

    :param hemisphere: string, typically either 'north' or 'south'
    :param date:
    :param bias_correct:
    :param source_path:
    """

    seas_file = os.path.join(
        source_path, hemisphere, "siconca",
        "{}.nc".format(date.replace(day=1).strftime("%Y%m%d")))

    if os.path.exists(seas_file):
        seas_da = xr.open_dataset(seas_file).siconc
    else:
        logging.warning("No SEAS data available at {}".format(seas_file))
        return None

    if bias_correct:
        # Let's have some maximum, though it's quite high
        (start_date, end_date) = (date - dt.timedelta(days=10 * 365),
                                  date + dt.timedelta(days=10 * 365))
        obs_da = get_obs_da(hemisphere, start_date, end_date)
        seas_hist_files = dict(
            sorted({
                os.path.abspath(el):
                    dt.datetime.strptime(os.path.basename(el)[0:8], "%Y%m%d")
                for el in glob.glob(
                    os.path.join(source_path, hemisphere, "siconca", "*.nc"))
                if re.search(r'^\d{8}\.nc$', os.path.basename(el)) and
                el != seas_file
            }.items()))

        def strip_overlapping_time(ds):
            data_file = os.path.abspath(ds.encoding["source"])

            try:
                idx = list(seas_hist_files.keys()).index(data_file)
            except ValueError:
                logging.exception("\n{} not in \n\n{}".format(
                    data_file, seas_hist_files))
                return None

            if idx < len(seas_hist_files) - 1:
                max_date = seas_hist_files[
                               list(seas_hist_files.keys())[idx + 1]] \
                           - dt.timedelta(days=1)
                logging.debug("Stripping {} to {}".format(data_file, max_date))
                return ds.sel(time=slice(None, max_date))
            else:
                logging.debug("Not stripping {}".format(data_file))
                return ds

        hist_da = xr.open_mfdataset(seas_hist_files,
                                    preprocess=strip_overlapping_time).siconc
        debiaser = LinearScaling(delta_type="additive",
                                 variable="siconc",
                                 reasonable_physical_range=[0., 1.])

        logging.info("Debiaser input ranges: obs {:.2f} - {:.2f}, "
                     "hist {:.2f} - {:.2f}, fut {:.2f} - {:.2f}".format(
                         float(obs_da.min()), float(obs_da.max()),
                         float(hist_da.min()), float(hist_da.max()),
                         float(seas_da.min()), float(seas_da.max())))

        seas_array = debiaser.apply(obs_da.values, hist_da.values,
                                    seas_da.values)
        seas_da.values = seas_array
        logging.info("Debiaser output range: {:.2f} - {:.2f}".format(
            float(seas_da.min()), float(seas_da.max())))

    logging.info("Returning SEAS data from {} from {}".format(seas_file, date))

    # This isn't great looking, but we know we're not dealing with huge
    # indexes in here
    date_location = list(seas_da.time.values).index(pd.Timestamp(date))
    if date_location > 0:
        logging.warning("SEAS forecast started {} day before the requested "
                        "date {}, make sure you account for this!".format(
                            date_location, date))

    seas_da = seas_da.sel(time=slice(date, None))
    logging.debug("SEAS data range: {} - {}, {} dates".format(
        pd.to_datetime(min(seas_da.time.values)).strftime("%Y-%m-%d"),
        pd.to_datetime(max(seas_da.time.values)).strftime("%Y-%m-%d"),
        len(seas_da.time)))

    return seas_da


def get_forecast_ds(forecast_file: object,
                    forecast_date: str,
                    stddev: bool = False) -> object:
    """

    :param forecast_file: a path to a .nc file
    :param forecast_date: initialisation date of the forecast
    :param stddev:
    :returns tuple(fc_ds, obs_ds, land_mask):
    """
    forecast_date = pd.to_datetime(forecast_date)

    forecast_ds = xr.open_dataset(forecast_file, decode_coords="all")
    get_key = "sic_mean" if not stddev else "sic_stddev"

    forecast_ds = getattr(
        forecast_ds.sel(time=slice(forecast_date, forecast_date)), get_key)

    return forecast_ds


def filter_ds_by_obs(ds: object, obs_da: object, forecast_date: str) -> object:
    """

    :param ds:
    :param obs_da:
    :param forecast_date: initialisation date of the forecast
    :return:
    """
    forecast_date = pd.to_datetime(forecast_date)
    (start_date,
     end_date) = (forecast_date + dt.timedelta(days=int(ds.leadtime.min())),
                  forecast_date + dt.timedelta(days=int(ds.leadtime.max())))

    if len(obs_da.time) < len(ds.leadtime):
        if len(obs_da.time) < 1:
            raise RuntimeError("No observational data available between {} "
                               "and {}".format(start_date.strftime("%D"),
                                               end_date.strftime("%D")))

        logging.warning("Observational data not available for full range of "
                        "forecast lead times: {}-{} vs {}-{}".format(
                            obs_da.time.to_series()[0].strftime("%D"),
                            obs_da.time.to_series()[-1].strftime("%D"),
                            start_date.strftime("%D"), end_date.strftime("%D")))
        (start_date, end_date) = (obs_da.time.to_series()[0],
                                  obs_da.time.to_series()[-1])

    # We broadcast to get a nicely compatible dataset for plotting
    return broadcast_forecast(start_date=start_date,
                              end_date=end_date,
                              dataset=ds)


def get_obs_da(
        hemisphere: str,
        start_date: str,
        end_date: str,
        obs_source: object = os.path.join(".", "data", "osisaf"),
) -> object:
    """

    :param hemisphere: string, typically either 'north' or 'south'
    :param start_date:
    :param end_date:
    :param obs_source:
    :return:
    """
    obs_years = pd.Series(pd.date_range(start_date, end_date)).dt.year.unique()
    obs_dfs = [
        el for yr in obs_years for el in glob.glob(
            os.path.join(obs_source, hemisphere, "siconca", "{}.nc".format(yr)))
    ]

    if len(obs_dfs) < len(obs_years):
        logging.warning(
            "Cannot find all obs source files for {} - {} in {}".format(
                start_date, end_date, obs_source))

    logging.info("Got files: {}".format(obs_dfs))
    obs_ds = xr.open_mfdataset(obs_dfs)
    obs_ds = obs_ds.sel(time=slice(start_date, end_date))

    return obs_ds.ice_conc


def get_crs(crs_str: str):
    """Get Coordinate Reference System (CRS) from string input argument

    Args:
        crs_str: A CRS given as EPSG code (e.g. `EPSG:3347` for North Canada)
            or, a pre-defined Cartopy CRS call (e.g. "PlateCarree")
    """
    if crs_str.casefold().startswith("epsg"):
        crs = ccrs.epsg(int(crs_str.split(":")[1]))
    elif crs_str == "Mercator.GOOGLE":
        crs = ccrs.Mercator.GOOGLE
    else:
        try:
            crs = getattr(ccrs, crs_str)()
        except AttributeError:
            get_crs_options = [crs_option for crs_option in dir(ccrs)
                                if isinstance(getattr(ccrs, crs_option), type)
                                 and issubclass(getattr(ccrs, crs_option), ccrs.CRS)
                                 ] + ["Mercator.GOOGLE"]
            get_crs_options.sort()
            get_crs_options = ", ".join(get_crs_options)
            raise AttributeError("Unsupported CRS defined, supported options are:",\
                f"{get_crs_options}"
            )

    return crs


def calculate_extents(x1: int, x2: int, y1: int, y2: int):
    """

    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :return:
    """
    data_extent_base = 5387500

    extents = [
        -data_extent_base + (x1 * 25000),
        data_extent_base - ((432 - x2) * 25000),
        -data_extent_base + (y1 * 25000),
        data_extent_base - ((432 - y2) * 25000),
    ]

    logging.debug("Data extents: {}".format(extents))
    return extents


# Convert pixel coordinates to projection coordinates
def pixel_to_projection(pixel_x_min, pixel_x_max,
                        pixel_y_min, pixel_y_max,
                        x_min_proj: float=-5387500, x_max_proj: float=5387500,
                        y_min_proj: float=-5387500, y_max_proj: float=5387500,
                        image_width: int=432, image_height: int=432,
                        ):
    """Converts pixel coordinates to CRS projection coordinates"""
    proj_x_min = (pixel_x_min / image_width ) * (x_max_proj - x_min_proj) + x_min_proj
    proj_x_max = (pixel_x_max / image_width ) * (x_max_proj - x_min_proj) + x_min_proj
    proj_y_min = (pixel_y_min / image_height) * (y_max_proj - y_min_proj) + y_min_proj
    proj_y_max = (pixel_y_max / image_height) * (y_max_proj - y_min_proj) + y_min_proj

    return proj_x_min, proj_x_max, proj_y_min, proj_y_max


def get_bounds(proj=None, pole=1):
    """Get min/max bounds for a given CRS projection"""
    if proj is None or isinstance(proj, ccrs.LambertAzimuthalEqualArea):
        proj = ccrs.LambertAzimuthalEqualArea(0, pole * 90)
        x_min_proj, x_max_proj = [-5387500, 5387500]
        y_min_proj, y_max_proj = [-5387500, 5387500]
    else:
        x_min_proj, x_max_proj = proj.x_limits
        y_min_proj, y_max_proj = proj.y_limits
    logging.debug(f"Projection bounds: {proj.x_limits}, {proj.y_limits}")
    return proj, x_min_proj, x_max_proj, y_min_proj, y_max_proj


def get_plot_axes(x1: int = 0,
                  x2: int = 432,
                  y1: int = 0,
                  y2: int = 432,
                  do_coastlines: bool = True,
                  north: bool = True,
                  south: bool = False,
                  proj = None,
                  set_extents: bool = False
                  ):
    """

    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :param do_coastlines:
    :param north:
    :param south:
    :param proj:
    :param method:
    :return:
    """
    assert north ^ south, "One hemisphere only must be selected"

    fig = plt.figure(figsize=(10, 8), dpi=150, layout='tight')

    if do_coastlines:
        pole = 1 if north else -1
        proj, x_min_proj, x_max_proj, y_min_proj, y_max_proj = get_bounds(proj, pole)

        ax = fig.add_subplot(1, 1, 1, projection=proj)

        extents = pixel_to_projection(x1, x2, y1, y2, x_min_proj, x_max_proj, y_min_proj, y_max_proj, 432, 432)

        ax.set_extent(extents, crs=proj)

        # Set colour for areas outside of `process_regions()` - no data here.
        ax.set_facecolor('dimgrey')
    else:
        ax = fig.add_subplot(1, 1, 1)

    return ax


def show_img(ax,
             arr,
             x1: int = 0,
             x2: int = 432,
             y1: int = 0,
             y2: int = 432,
             cmap: object = None,
             do_coastlines: bool = True,
             vmin: float = 0.,
             vmax: float = 1.,
             north: bool = True,
             south: bool = False,
             crs: object = None,
             extents: list = None
             ):
    """

    :param ax:
    :param arr:
    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :param cmap:
    :param do_coastlines:
    :param vmin:
    :param vmax:
    :param north:
    :param south:
    :return:
    """

    assert north ^ south, "One hemisphere only must be selected"

    if do_coastlines:
        pole = 1 if north else -1
        data_crs = ccrs.LambertAzimuthalEqualArea(0, pole * 90)
        extents = calculate_extents(x1, x2, y1, y2)
        im = ax.imshow(arr,
                       vmin=vmin,
                       vmax=vmax,
                       cmap=cmap,
                       transform=data_crs,
                       extent=extents)
        ax.coastlines()
    else:
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)

    return im


def process_probes(probes, data) -> tuple:
    """
    :param probes: A sequence of locations (pairs)
    :param data: A sequence of xr.DataArray
    """

    # index into each element of data with a xr.DataArray, for pointwise
    # selection.  Construct the indexing DataArray as follows:

    probes_da = xr.DataArray(probes, dims=('probe', 'coord'))
    xcs, ycs = probes_da.sel(coord=0), probes_da.sel(coord=1)

    for idx, arr in enumerate(data):
        arr = arr.assign_coords({
            "xi": ("xc", np.arange(len(arr.xc))),
            "yi": ("yc", np.arange(len(arr.yc))),
        })
        if arr is not None:
            data[idx] = arr.isel(xc=xcs, yc=ycs)

    return data


def reproject_projected_coords(data, target_crs=ccrs.Mercator(), pole=1):
    if pole == 1:
        data_crs_proj = ccrs.NorthPolarStereo()
    elif pole == -1:
        data_crs_proj = ccrs.SouthPolarStereo()
    data_crs_geo = ccrs.PlateCarree()

    x_m, y_m = data.xc.values*1000, data.yc.values*1000
    transformed_coords_proj = target_crs.transform_points(data_crs_proj, x_m, y_m)

    trans_x = transformed_coords_proj[..., 0]
    trans_y = transformed_coords_proj[..., 1]

    lon, lat = data.lon.values, data.lat.values
    transformed_coords_proj = target_crs.transform_points(data_crs_geo, lon, lat)

    trans_lon = transformed_coords_proj[..., 0]
    trans_lat = transformed_coords_proj[..., 1]

    data_crs = ccrs.LambertAzimuthalEqualArea(0, 90)
    # target_crs = ccrs.epsg("3347")
    target_crs = ccrs.Mercator()
    target_crs = ccrs.Mercator.GOOGLE
    # target_crs = ccrs.PlateCarree()
    # target_crs = data_crs

    data_reproject = xr.DataArray(
        data.data,
                dims=["time", "leadtime", "y", "x"],
                coords={
                    "x": data.xc.data*1000,
                    "y": data.yc.data*1000,
                    # "lon": (("y", "x"), data.lon.data),
                    # "lat": (("y", "x"), data.lat.data),
                    "time": data.time.data,
                    "leadtime": data.leadtime.data,
                }
    )

    data_reproject.rio.set_spatial_dims(x_dim="y", y_dim="x", inplace=True)
    data_reproject.rio.write_crs(data_crs.proj4_init, inplace=True)
    data_reproject.rio.write_nodata(np.nan, inplace=True)

    # Reproject to Mercator
    data_mercator = data_reproject.isel(time=0, leadtime=0).rio.reproject(target_crs.proj4_init,
        # resampling=Resampling.bilinear,
        nodata=np.nan
        )

    # Compute lat/lon for reprojected image
    data_geo = data_mercator.rio.reproject(data_crs_geo.proj4_init, shape=data_mercator.shape)    
    lon_grid, lat_grid = np.meshgrid(data_geo.x, data_geo.y)

    data_mercator["lon"] = (("y", "x"), lon_grid)
    data_mercator["lat"] = (("y", "x"), lat_grid)

    # Define your pixel bounds
    min_x_pixel = 10
    max_x_pixel = 150
    min_y_pixel = 20
    max_y_pixel = 200

    x_max, y_max = data_mercator.x.shape[0], data_mercator.y.shape[0]
    max_x_pixel = min(x_max, max_x_pixel)
    max_y_pixel = min(y_max, max_y_pixel)

    # Clip the data array
    clipped_data = data_mercator[..., (y_max - max_y_pixel):(y_max - min_y_pixel), min_x_pixel:max_x_pixel]

    # plt.figure(figsize=(10, 10))
    # ax = plt.axes(projection=target_crs)
    # clipped_data.plot.imshow(ax=ax, transform=target_crs)
    # # ax.imshow(clipped_data, transform=ccrs.Mercator.GOOGLE)
    # ax.coastlines()
    # ax.set_global()
    # plt.show()

    return clipped_data


def process_regions(region: tuple,
        data: tuple,
        method: str = "pixel",
        proj=None,
        pole=1,
    ) -> tuple:
    """Extract subset of pan-Arctic/Antarctic region based on region bounds.

    :param region: Either image pixel bounds, or lat/lon bounds.
    :param data: Contains the full xarray DataArray.
    :param method: Whether providing pixel coordinates or lat/lon.

    :return:
    """

    assert len(region) == 4, "Region needs to be a list of four integers"
    x1, y1, x2, y2 = region
    assert x2 > x1 and y2 > y1, "Region is not valid"

    for idx, arr in enumerate(data):
        if arr is not None:
            arr = reproject_projected_coords(arr)

            if method == "pixel":
                data[idx] = arr[..., (432 - y2):(432 - y1), x1:x2]

                # proj, x_min_proj, x_max_proj, y_min_proj, y_max_proj = get_bounds(proj, pole)
                # x_min, x_max, y_min, y_max = pixel_to_projection(x1, x2, y1, y2, x_min_proj, x_max_proj, y_min_proj, y_max_proj, 432, 432)
                # data[idx] = arr.sel(xc=slice(x_min/1000, x_max/1000), yc=slice(y_max/1000, y_min/1000))
            elif method == "lat_lon":
                # Create condition where data is within lat/lon region
                condition = (arr.lat >= x1) & (arr.lat <= x2) & (arr.lon >= y1) & (arr.lon <= y2)

                # Extract subset within region using where()
                data[idx] = arr.where(condition.compute(), drop=True)
            else:
                raise NotImplementedError

    return data


@cache
def lat_lon_box(lon_bounds: np.array, lat_bounds: np.array, segments: int=1):
    """Rectangular boundary coordinates in lat/lon coordinates.

    Args:
        lon_bounds: (min, max) lon values
        lat_bounds: (min, max) lat values
        segments: Number of segments per edge

    Returns:
        (lats, lons) for rectangular boundary region
    """

    segments += 1
    rectangular_sides = 4

    lats = np.empty((segments*rectangular_sides))
    lons = np.empty((segments*rectangular_sides))

    bounds = [
        [0, 0],
        [-1, 0],
        [-1, -1],
        [0, -1],
    ]

    for i, (lat_min, lat_max) in enumerate(bounds):
        lats[i*segments:(i+1)*segments] = np.linspace(lat_bounds[lat_min], lat_bounds[lat_max], num=segments)

    bounds.reverse()

    for i, (lon_min, lon_max) in enumerate(bounds):
        lons[i*segments:(i+1)*segments] = np.linspace(lon_bounds[lon_min], lon_bounds[lon_max], num=segments)

    return lats, lons

def get_custom_cmap(cmap, vmin=0, vmax=1):
    """Creates a new colormap for valid array with range 0-1, but with nan set to <0.

    Hack since cartopy needs transparency for nan regions to wraparound
        correctly with pcolormesh.
    """
    colors = cmap(np.linspace(vmin, vmax, cmap.N))
    custom_cmap = mpl.colors.ListedColormap(colors)
    custom_cmap.set_bad("dimgrey", alpha=0)
    custom_cmap.set_under("dimgrey")
    return custom_cmap
