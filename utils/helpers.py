from shapely.ops import cascaded_union, polygonize
import shapely.geometry as geometry
from scipy.spatial import Delaunay
import numpy as np
import pyresample as pyr
import pandas as pd
import geopandas as gpd
import re
from shapely.geometry import Point
from fiona.crs import from_epsg
import dask.array as da


def get_epsg_from_string(string):
    pattern = r'[-+]?\d+'
    epsg = int(re.findall(pattern, string)[0])
    return epsg


def geopandas_to_numpy(geometries):
    return np.array([[geom.xy[0][0], geom.xy[1][0]] for geom in geometries])


def xarray_to_epsg(xset, epsg):
    this_epsg = get_epsg_from_string(xset.attrs['crs'])

    if epsg == this_epsg:
        return xset

    ptsx = [Point(x, xset.coords['y'].data[0]) for x in xset.coords['x'].data]
    ptsy = [Point(xset.coords['x'].data[0], y) for y in xset.coords['y'].data]
    pts = ptsx + ptsy

    df = gpd.GeoDataFrame({'geometry': pts}, crs=from_epsg(this_epsg))
    df = df.to_crs(epsg=epsg)

    x = [p.x for p in df.geometry[:len(ptsx)]]
    y = [p.y for p in df.geometry[len(ptsx):]]
    xset.coords['x'] = list(x)
    xset.coords['y'] = list(y)

    xset.attrs['crs']='+init=epsg:' + str(epsg)
    return xset


def binarize_dataframe(dframe, var, vals, pad_lo=None, pad_hi=None):
    if pad_lo is not None:
        vals = np.concatenate((np.array([pad_lo]), vals))
    if pad_hi is not None:
        vals = np.concatenate((vals, np.array([pad_hi])))

    bin_edges = vals[:-1] + (vals[1:] - vals[:-1]) / 2
    idxs = pd.cut(dframe[var], bins=bin_edges, labels=False)

    _binned_dframes = {}
    for i in np.unique(idxs):
        dfi = dframe.iloc[np.where(idxs == i)[0]]
        _binned_dframes[i] = dfi

    return _binned_dframes


def raster_to_point(dframe, xset, method='nearest', radius_of_influence=1000, inplace=True, *args, **kwargs):
    resample_funcs = {'nearest': pyr.kd_tree.resample_nearest}
    resample_func = resample_funcs[method]

    grid = None

    # for non time dependent vars
    swath = None

    # for time_mode
    _time_binned_swaths = {}
    _time_binned_dframes = {}

    if not inplace:
        dframe = dframe.copy()

    # make sure we are in lat / lon coordinates which is what pyresample assumes
    assert get_epsg_from_string(dframe.crs['init']) == 4326

    # make sure we are in lat / lon coordinates which is what pyresample assumes
    # since all vars in xset are aligned, check only first one
    # fixme: is this correct?
    first = list(xset.items())[0][1]
    xset_epsg = get_epsg_from_string(first.attrs['crs'])
    if not xset_epsg == 4326:
        xset = xarray_to_epsg(xset, 4326)

    time_deps = {}
    time_indeps = []
    for var in xset.data_vars:
        dframe[var] = np.nan

        # all vars in xset are aligned
        if grid is None:
            lats = xset[var]['y']
            lons = xset[var]['x']
            lons, lats = np.meshgrid(lons, lats)
            grid = pyr.geometry.GridDefinition(lats=lats, lons=lons)

        if 'time' in xset[var].coords:
            # all vars in xset have same time resolution, so create index and swaths only once
            if _time_binned_swaths == {}:
                _time_binned_dframes = binarize_dataframe(dframe, 'time', xset[var].coords['time'].data,
                                                          pad_lo=np.datetime64('1970-01-01'),
                                                          pad_hi=np.datetime64('now'))

                _time_binned_swaths = {i: pyr.geometry.SwathDefinition(lats=dfi.y, lons=dfi.x)
                                       for i, dfi in _time_binned_dframes.items()}

            # prepare column for time_delta
            dframe[var + '_time_delta'] = np.nan
            dframe[var + '_time_ind'] = np.nan

            for i in _time_binned_dframes.keys():
                if i in time_deps:
                    time_deps[i][var] = xset[var].isel(time=i).data
                else:
                    time_deps[i] = {var: xset[var].isel(time=i).data}

        # if variable is not time dependent, create one entry with all data points
        else:
            # same data points for all vars
            if swath is None:
                swath = pyr.geometry.SwathDefinition(lats=dframe.y, lons=dframe.x)

            time_indeps.append((var, xset[var].data))

    # If there are time dependent vars, run kdtree queries including time independent vars
    if time_deps != {}:
        for i in time_deps:
            # stack arrays
            time_dep_vars, time_dep_arrs = zip(*time_deps[i].items())

            if time_indeps:
                time_indep_vars, time_indep_arrs = zip(*time_indeps)
            else:
                time_indep_vars, time_indep_arrs = [], []

            arrs = list(time_dep_arrs) + list(time_indep_arrs)
            if len(arrs) > 1:
                data_at_t = da.stack([np.array(a) for a in arrs], axis=2)
                data_at_t = np.array(data_at_t)
            else:
                data_at_t = np.array(arrs[0])

            data = resample_func(source_geo_def=grid,
                                 target_geo_def=_time_binned_swaths[i],
                                 data=data_at_t,
                                 radius_of_influence=radius_of_influence,
                                 fill_value = np.nan,
                                 *args, **kwargs)

            if len(data_at_t.shape) == 2:
                data = np.atleast_2d(data).transpose()

            dfi = _time_binned_dframes[i]

            # assign time dependent vars to dframe
            for i, var in enumerate(time_dep_vars):
                dframe.loc[dfi.index, var] = pd.Series(data[:, i])

                xset_time = xset.coords['time'][i].data
                time_delta = dfi.time.astype(np.datetime64).subtract(xset_time)

                dframe.loc[dfi.index, var + '_time_delta'] = time_delta
                dframe.loc[dfi.index, var + '_time_ind'] = np.int(i)

            # assign time independent vars to dframe
            for i, var in enumerate(time_indep_vars):
                dframe.loc[dfi.index, var] = pd.Series(data[:, i+len(time_dep_vars)])
        return dframe

    # if there are no time independent vars, run kdtree query for all points at once
    if time_indeps:
        time_indep_vars, time_indep_arrs = zip(*time_indeps)
        arrs = list(time_indep_arrs)

        print(len(arrs), arrs)
        if len(arrs) > 1:
            vars = da.stack([np.array(a) for a in arrs], axis=2)
            vars = np.array(vars)
        else:
            vars = np.array(arrs[0])

        data = resample_func(source_geo_def=grid,
                             target_geo_def=swath,
                             data=vars,
                             radius_of_influence=radius_of_influence,
                             *args, **kwargs)

        if len(vars.shape) == 2:
            data = np.atleast_2d(data).transpose()

        for i, var in enumerate(time_indep_vars):
            dframe.loc[:, var] = pd.Series(data[:, i])

    return dframe


def concave_hull(points, alpha):
    """
    from https://gist.github.com/dwyerk/10561690

    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]

    a = ((triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2 + (triangles[:, 0, 1] - triangles[:, 1, 1]) ** 2) ** 0.5
    b = ((triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2 + (triangles[:, 1, 1] - triangles[:, 2, 1]) ** 2) ** 0.5
    c = ((triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2 + (triangles[:, 2, 1] - triangles[:, 0, 1]) ** 2) ** 0.5

    s = (a + b + c) / 2.0
    areas = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    circums = a * b * c / (4.0 * areas)

    filtered = triangles[circums < (1.0 / alpha)]

    edge1 = filtered[:, (0, 1)]
    edge2 = filtered[:, (1, 2)]
    edge3 = filtered[:, (2, 0)]
    edge_points = np.unique(np.concatenate((edge1, edge2, edge3)), axis=0).tolist()

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))

    return cascaded_union(triangles), edge_points
