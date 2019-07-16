from shapely.ops import cascaded_union, polygonize
import shapely.geometry as geometry
from scipy.spatial import Delaunay
import numpy as np
import pyresample as pyr
import pandas as pd
import re


def get_epsg_from_geopandas(dframe):
    pattern = r'[-+]?\d+'
    epsg = int(re.findall(pattern, dframe.crs['init'])[0])
    return epsg


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
    assert get_epsg_from_geopandas(dframe) == 4326

    # make sure we are in lat / lon coordinates which is what pyresample assumes
    # since all vars in xset are aligned, check only first one
    assert list(xset.items())[0][1].attrs['crs'] == '+init=epsg:4326'

    for var in xset.data_vars:

        if 'time' in xset[var].coords:
            is_time_dependent_var = True
        else:
            is_time_dependent_var = False

        # all vars in xset are aligned
        if grid is None:
            lats = xset[var]['y']
            lons = xset[var]['x']
            lons, lats = np.meshgrid(lons, lats)
            grid = pyr.geometry.GridDefinition(lats=lats, lons=lons)

        if is_time_dependent_var:
            # all vars in xset have same time resolution, so create index and swaths only once
            if _time_binned_swaths == {}:
                _time_binned_dframes = binarize_dataframe(dframe, 'time', xset[var].coords['time'].data,
                                                          pad_lo=np.datetime64('1970-01-01'),
                                                          pad_hi=np.datetime64('now'))

                _time_binned_swaths = {i: pyr.geometry.SwathDefinition(dfi.y, dfi.x)
                                       for i, dfi in _time_binned_dframes.items()}

            # prepare column for time_delta
            dframe[var + '_time_delta'] = np.nan
            dframe[var + '_time_ind'] = np.nan

            times = [(xset[var].isel(time=i),
                      _time_binned_dframes[i],
                      _time_binned_swaths[i],
                      i)
                     for i in _time_binned_swaths.keys()]

        # if variable is not time dependent, create one entry with all data points
        else:
            # same data points for all vars
            if swath is None:
                swath = pyr.geometry.SwathDefinition(lats=dframe.y, lons=dframe.x)

            times = [(xset[var], dframe, swath, None)]

        # Iterate over all times
        dframe[var] = np.nan
        for var_at_t, dframe_at_t, swath_at_t, time_ind in times:
            if dframe_at_t.empty:
                continue

            data = resample_func(source_geo_def=grid,
                                 target_geo_def=swath_at_t,
                                 data=var_at_t.data,
                                 radius_of_influence=radius_of_influence,
                                 *args, **kwargs)

            dframe.loc[dframe_at_t.index, var] = pd.Series(data)
            if is_time_dependent_var:
                xset_time = xset.coords['time'][time_ind].data
                time_delta = dframe_at_t.time.astype(np.datetime64).subtract(xset_time)

                dframe.loc[dframe_at_t.index,
                           var + '_time_delta'] = time_delta
                dframe.loc[dframe_at_t.index,
                           var + '_time_ind'] = time_ind
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
