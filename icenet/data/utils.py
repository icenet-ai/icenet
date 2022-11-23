import logging
import requests

import cartopy.crs as ccrs
import iris
import numpy as np


def assign_lat_lon_coord_system(cube: object):
    """Assign coordinate system to iris cube to allow regridding.

    :param cube:
    """

    # This originated from era5
    cube.coord('latitude').coord_system = iris.coord_systems.GeogCS(6367470.0)
    cube.coord('longitude').coord_system = iris.coord_systems.GeogCS(6367470.0)

    # NOTE: CMIP6 original assignment, but cs in this case is bring assigned to
    # a module call that doesn't exist, let alone contain this member
    # cs = grid_cube.coord_system().ellipsoid
    # for coord in ['longitude', 'latitude']:
    #     cmip6_cube.coord(coord).coord_system = cs

    return cube


def rotate_grid_vectors(u_cube: object,
                        v_cube: object,
                        angles: object):
    """
    Author: Tony Phillips (BAS)

    Wrapper for :func:`~iris.analysis.cartography.rotate_grid_vectors`
    that can rotate multiple masked spatial fields in one go by iterating
    over the horizontal spatial axes in slices

    :param u_cube:
    :param v_cube:
    :param angles:
    :return:

    """
    # lists to hold slices of rotated vectors
    u_r_all = iris.cube.CubeList()
    v_r_all = iris.cube.CubeList()

    # get the X and Y dimension coordinates for each source cube
    u_xy_coords = [u_cube.coord(axis='x', dim_coords=True),
                   u_cube.coord(axis='y', dim_coords=True)]
    v_xy_coords = [v_cube.coord(axis='x', dim_coords=True),
                   v_cube.coord(axis='y', dim_coords=True)]

    # iterate over X, Y slices of the source cubes, rotating each in turn
    for u, v in zip(u_cube.slices(u_xy_coords, ordered=False),
                    v_cube.slices(v_xy_coords, ordered=False)):
        u_r, v_r = iris.analysis.cartography.rotate_grid_vectors(u, v, angles)
        u_r_all.append(u_r)
        v_r_all.append(v_r)

    # return the slices, merged back together into a pair of cubes
    return u_r_all.merge_cube(), v_r_all.merge_cube()


def gridcell_angles_from_dim_coords(cube: object):
    """
    Author: Tony Phillips (BAS)

    Wrapper for :func:`~iris.analysis.cartography.gridcell_angles`
    that derives the 2D X and Y lon/lat coordinates from 1D X and Y
    coordinates identifiable as 'x' and 'y' axes

    The provided cube must have a coordinate system so that its
    X and Y coordinate bounds (which are derived if necessary)
    can be converted to lons and lats

    :param cube:
    :return:
    """

    # get the X and Y dimension coordinates for the cube
    x_coord = cube.coord(axis='x', dim_coords=True)
    y_coord = cube.coord(axis='y', dim_coords=True)

    # add bounds if necessary
    if not x_coord.has_bounds():
        x_coord = x_coord.copy()
        x_coord.guess_bounds()
    if not y_coord.has_bounds():
        y_coord = y_coord.copy()
        y_coord.guess_bounds()

    # get the grid cell bounds
    x_bounds = x_coord.bounds
    y_bounds = y_coord.bounds
    nx = x_bounds.shape[0]
    ny = y_bounds.shape[0]

    # make arrays to hold the ordered X and Y bound coordinates
    x = np.zeros((ny, nx, 4))
    y = np.zeros((ny, nx, 4))

    # iterate over the bounds (in order BL, BR, TL, TR), mesh them and
    # put them into the X and Y bound coordinates (in order BL, BR, TR, TL)
    c = [0, 1, 3, 2]
    cind = 0
    for yi in [0, 1]:
        for xi in [0, 1]:
            xy = np.meshgrid(x_bounds[:, xi], y_bounds[:, yi])
            x[:,:,c[cind]] = xy[0]
            y[:,:,c[cind]] = xy[1]
            cind += 1

    # convert the X and Y coordinates to longitudes and latitudes
    source_crs = cube.coord_system().as_cartopy_crs()
    target_crs = ccrs.PlateCarree()
    pts = target_crs.transform_points(source_crs, x.flatten(), y.flatten())
    lons = pts[:, 0].reshape(x.shape)
    lats = pts[:, 1].reshape(x.shape)

    # get the angles
    angles = iris.analysis.cartography.gridcell_angles(lons, lats)

    # add the X and Y dimension coordinates from the cube to the angles cube
    angles.add_dim_coord(y_coord, 0)
    angles.add_dim_coord(x_coord, 1)

    # if the cube's X dimension preceeds its Y dimension
    # transpose the angles to match
    if cube.coord_dims(x_coord)[0] < cube.coord_dims(y_coord)[0]:
        angles.transpose()

    return angles


def invert_gridcell_angles(angles: object):
    """
    Author: Tony Phillips (BAS)

    Negate a cube of gridcell angles in place, transforming
    gridcell_angle_from_true_east <--> true_east_from_gridcell_angle
    :param angles:
    """
    angles.data *= -1

    names = ['true_east_from_gridcell_angle', 'gridcell_angle_from_true_east']
    name = angles.name()
    if name in names:
        angles.rename(names[1 - names.index(name)])


def esgf_search(server: str = "https://esgf-node.llnl.gov/esg-search/search",
                files_type: str = "OPENDAP",
                local_node: bool = False,
                latest: bool = True,
                project: str = "CMIP6",
                format: str = "application%2Fsolr%2Bjson",
                use_csrf: bool = False,
                **search):
    """

    Below taken from
    https://hub.binder.pangeo.io/user/pangeo-data-pan--cmip6-examples-ro965nih/lab
    and adapted slightly

    :param server:
    :param files_type:
    :param local_node:
    :param latest:
    :param project:
    :param format:
    :param use_csrf:
    :param search:
    :return:
    """
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"]="File"
    if latest:
        payload["latest"] = "true"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if 'csrftoken' in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies['csrftoken']
        else:
            # older versions
            csrftoken = client.cookies['csrf']
        payload["csrfmiddlewaretoken"] = csrftoken

    payload["format"] = format

    offset = 0
    numFound = 10000
    all_files = []
    files_type = files_type.upper()
    while offset < numFound:
        payload["offset"] = offset
        url_keys = []
        for k in payload:
            url_keys += ["{}={}".format(k, payload[k])]

        url = "{}/?{}".format(server, "&".join(url_keys))
        logging.debug("ESGF search URL: {}".format(url))

        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        resp = resp["docs"]
        offset += len(resp)
        for d in resp:
            for k in d:
                logging.debug("{}: {}".format(k,d[k]))
            url = d["url"]
            for f in d["url"]:
                sp = f.split("|")
                if sp[-1] == files_type:
                    all_files.append(sp[0].split(".html")[0])
    return sorted(all_files)
