import datetime as dt
import glob
import logging
import os
import re

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.array as da
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

from functools import cache
from ibicus.debias import LinearScaling
from pyproj import CRS, Transformer
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
                  north: bool = True,
                  south: bool = False,
                  geoaxes: bool = True,
                  coastlines: str = None,
                  gridlines: bool = True,
                  target_crs: object = ccrs.Mercator(),
                  transform_crs: object = ccrs.PlateCarree(),
                  figsize: int = (10, 8),
                  dpi: int = 150,
                  ):
    """

    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :param geoaxes:
    :return:
    """
    assert north ^ south, "One hemisphere only must be selected"

    fig = plt.figure(figsize=figsize, dpi=dpi, layout="tight")

    if geoaxes:
        # pole = 1 if north else -1
        # target_crs, x_min_proj, x_max_proj, y_min_proj, y_max_proj = get_bounds(target_crs, pole)

        ax = fig.add_subplot(1, 1, 1, projection=target_crs)
        plt.tight_layout(pad=4.0)

        # extents = pixel_to_projection(x1, x2, y1, y2, x_min_proj, x_max_proj, y_min_proj, y_max_proj, 432, 432)
        # ax.set_extent(extents, crs=proj)

        # Set colour for areas outside of `process_regions()` - no data here.
        ax.set_facecolor('dimgrey')

        if coastlines is not None:
            ax.add_feature(cfeature.LAND, facecolor="dimgrey", zorder=1)
            if coastlines.casefold() == "gshhs":
                # Higher resolution coastlines when a region is specified
                ax.add_feature(cfeature.GSHHSFeature(scale="auto", levels=[1]), zorder=100)
            else:
                ax.coastlines(resolution="50m", zorder=100)

        if gridlines:
            gl = ax.gridlines(crs=transform_crs, draw_labels=True)
            # Prevent generating labels beneath the colourbar
            gl.top_labels = False
            gl.right_labels = False

    else:
        ax = fig.add_subplot(1, 1, 1)

    return fig, ax


def show_img(ax,
             arr,
             x1: int = 0,
             x2: int = 432,
             y1: int = 0,
             y2: int = 432,
             cmap: object = None,
             geoaxes: bool = True,
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
    :param geoaxes:
    :param vmin:
    :param vmax:
    :param north:
    :param south:
    :return:
    """

    assert north ^ south, "One hemisphere only must be selected"

    if geoaxes:
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


def reproject_array(array, target_crs):
    return array.rio.reproject(target_crs.proj4_init,
        # resampling=Resampling.bilinear,
        nodata=np.nan
        )

def process_block(block, target_crs):
    # dataarray = xr.DataArray(block, dims=["leadtime", "y", "x"])
    dataarray = block
    reprojected = reproject_array(dataarray, target_crs)
    return reprojected.drop_vars(["time"])


def reproject_projected_coords(data,
                                target_crs=ccrs.Mercator(),
                                pole=1,
                                ):
    # Eastings/Northings projection
    data_crs_proj = ccrs.LambertAzimuthalEqualArea(0, pole*90)
    # geographic projection
    data_crs_geo = ccrs.PlateCarree()

    data_reproject = data.copy()
    data_reproject = data_reproject.drop_vars(["Lambert_Azimuthal_Grid", "lon", "lat"])
    data_reproject = data_reproject.assign_coords({"xc": data_reproject.xc.data*1000,
                                    "yc": data_reproject.yc.data*1000
                                })

    # Need to use correctly scaled xc and yc to get coastlines working even if not reprojecting.
    # So, just return scaled DataArray back and not reproject if don't need to.
    if target_crs == data_crs_proj:
        return data_reproject

    # Set xc, yc (eastings and northings) projection details
    data_reproject = data_reproject.rename({"xc": "x", "yc": "y"})
    data_reproject.rio.write_crs(data_crs_proj.proj4_init, inplace=True)
    data_reproject.rio.write_nodata(np.nan, inplace=True)

    times = len(data_reproject.time)
    leadtimes = len(data_reproject.leadtime)

    # Create a sample image block for use as template for Dask
    sample_block = data_reproject.isel(time=0, leadtime=0)
    sample_reprojected =  reproject_array(sample_block, target_crs)

    # Create a template DataArray based on the reprojected sample block
    template_shape = (data_reproject.sizes['leadtime'], sample_reprojected.sizes['y'], sample_reprojected.sizes['x'])
    template_data = da.zeros(template_shape, chunks=(1, -1, -1))
    template = xr.DataArray(template_data, dims=['leadtime', 'y', 'x'],
                            coords={'leadtime': data_reproject.coords['leadtime'],
                            'y': sample_reprojected.coords['y'],
                            'x': sample_reprojected.coords['x'],
                            }
                            )

    reprojected_data = []
    for time in range(times):
        leadtime_data = xr.map_blocks(process_block, data_reproject.isel(time=time), template=template, kwargs={"target_crs": target_crs})
        reprojected_data.append(leadtime_data)

    # TODO: Add projection info into DataArray, like the `Lambert_Azimuthal_Grid` dropped above
    reprojected_data = xr.concat(reprojected_data, dim="time")
    reprojected_data.coords["time"] = data_reproject.time.data

    # Set attributes
    reprojected_data.rio.write_crs(target_crs.proj4_init, inplace=True)
    reprojected_data.rio.write_nodata(np.nan, inplace=True)

    # Compute geographic for reprojected image
    transformer = Transformer.from_crs(target_crs.proj4_init, data_crs_geo.proj4_init)
    x = reprojected_data.x.values
    y = reprojected_data.y.values

    X, Y = np.meshgrid(x, y)
    lon_grid, lat_grid = transformer.transform(X, Y)

    reprojected_data["lon"] = (("y", "x"), lon_grid)
    reprojected_data["lat"] = (("y", "x"), lat_grid)

    # Rename back to 'xc' and 'yc', although, these are now in metres rather than 1000 metres
    reprojected_data = reprojected_data.rename({"x": "xc", "y": "yc"})

    return reprojected_data


def process_regions(region: tuple=None,
        data: tuple=None,
        method: str = "pixel",
        target_crs=ccrs.Mercator.GOOGLE,
        pole=1,
        clip_geographic_region=True,
    ) -> tuple:
    """Extract subset of pan-Arctic/Antarctic region based on region bounds.

    :param region: Either image pixel bounds, or geographic bounds.
    :param data: Contains the full xarray DataArray.
    :param method: Whether providing pixel coordinates or geographic (i.e. lon/lat).
    :param clip_geographic_region: Whether to clip the data to the defined lon/lat region bounds.

    :return:
    """

    if region is not None:
        assert len(region) == 4, "Region needs to be a list of four integers"
        x1, y1, x2, y2 = region
        assert x2 > x1 and y2 > y1, "Region is not valid"
        if method == "geographic":
            assert x1 >= -180 and x2 <= 180, "Expect longitude range to be `-180<=longitude>=180`"

    for idx, arr in enumerate(data):
        if arr is not None:
            if (method == "geographic" and clip_geographic_region):
                # Reproject when region is bounded by lon/lat without the 'clip_geographic_region' flag
                data[idx] = arr
            else:
                logging.info(f"Reprojecting data to specified CRS")
                reprojected_data = reproject_projected_coords(arr,
                            target_crs=target_crs,
                            pole=pole,
                            )
                data[idx] = reprojected_data


            if region is not None:
                logging.info(f"Clipping data to specified bounds: {region}")
                if method.casefold() == "geographic":
                    if clip_geographic_region:
                        # Limit to lon/lat region, within a given tolerance
                        tolerance = 1E-1
                        # Create condition where data is within geographic (lon/lat) region
                        condition = (arr.lon >= x1-tolerance) & (arr.lon <= x2+tolerance) & \
                                    (arr.lat >= y1-tolerance) & (arr.lat <= y2+tolerance)

                        # Extract subset within region using where()
                        clipped_data = arr.where(condition, drop=True)

                        # Reproject just the clipped region for speed
                        data[idx] = reproject_projected_coords(clipped_data,
                                                    target_crs=target_crs,
                                                    pole=pole,
                                                    )
                elif method.casefold() == "pixel":
                    x_max, y_max = reprojected_data.xc.shape[0], reprojected_data.yc.shape[0]

                    # Clip the data array to specified pixel region
                    data[idx] = reprojected_data[..., (y_max - y2):(y_max - y1), x1:x2]
                else:
                    raise NotImplementedError("Only method='pixel' or 'geographic' bounds are supported")

    return data


@cache
def geographic_box(lon_bounds: np.array, lat_bounds: np.array, segments: int=1):
    """Rectangular boundary coordinates in lon/lat coordinates.

    Args:
        lon_bounds: (min, max) lon values
        lat_bounds: (min, max) lat values
        segments: Number of segments per edge

    Returns:
        (lats, lons) for rectangular boundary region
    """

    segments += 1
    rectangular_sides = 4

    lons = np.empty((segments*rectangular_sides))
    lats = np.empty((segments*rectangular_sides))

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

    return lons, lats

def get_custom_cmap(cmap):
    """Creates a new colormap, but with nan set to <0.

    Hack since cartopy needs transparency for nan regions to wraparound
        correctly with pcolormesh.
    """
    colors = cmap(np.linspace(0, 1, cmap.N))
    custom_cmap = mpl.colors.ListedColormap(colors)
    custom_cmap.set_bad("dimgrey", alpha=0)
    custom_cmap.set_under("dimgrey")
    return custom_cmap
