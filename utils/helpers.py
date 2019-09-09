from shapely.ops import cascaded_union, polygonize
import shapely.geometry as geometry
from scipy.spatial import Delaunay
import numpy as np
import pandas as pd
import geopandas as gpd
import re
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

import rasterstats as rstats
from fiona.crs import from_epsg

from joblib import Parallel, delayed


def get_epsg_from_string(string):
    pattern = r'[-+]?\d+'
    epsg = int(re.findall(pattern, string)[0])
    return epsg


def to_crs(dframe, crs=None, epsg=None, inplace=False):
    assert not (crs is None and epsg is None)
    if epsg is not None:
        crs = from_epsg(epsg)
    if not inplace:
        dframe = dframe.to_crs(crs)
    else:
        dframe.to_crs(crs, inplace=True)

    xy = np.array([[pt.coords[0][0], pt.coords[0][1]] for pt in dframe['geometry']])
    dframe['x'] = xy[:, 0]
    dframe['y'] = xy[:, 1]
    return dframe


def geopandas_to_numpy(geometries):
    return np.array([[geom.xy[0][0], geom.xy[1][0]] for geom in geometries])


def xarray_to_epsg(xset, epsg):
    this_epsg = get_epsg_from_string(xset.attrs['crs']['init'])

    if epsg is None or epsg == this_epsg:
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

    xset.attrs['crs'] = dict(from_epsg(epsg))

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


def raster_to_point(dframe, xset, inplace=True, n_jobs=1, interpolate='nearest', zonal_stats=False, buffer=None,
                    add_stats=None, col_name='', *args, **kwargs):

    assert not (zonal_stats and buffer is None)

    time_indeps, time_deps, time_binned_dframes = _prep_raster_to_point(dframe, xset)

    if not inplace:
        dframe = dframe.copy()

    if add_stats is None:
        add_stats = {}

    def point_query(df, path, crs, *args, **kwargs):
        if not df.crs == crs:
            df = df.to_crs(crs)
        return rstats.point_query(df.geometry, path, interpolate=interpolate, *args, **kwargs)

    def zonal_query(df, path, crs, *args, **kwargs):
        geoms = gpd.GeoDataFrame(geometry=[geom.buffer(buffer) for geom in df.geometry], crs=df.crs)
        if not df.crs == crs:
            geoms = geoms.to_crs(crs)
        return rstats.zonal_stats(geoms, path, add_stats=add_stats)

    if zonal_stats:
        query_func = zonal_query
        if col_name == '':
            col_name = 'zonal'
    else:
        query_func = point_query

    if time_deps != {}:
        # @fixme: might be useful to pass dfi= time_binned_dframes[time_idx].copy() as subsequent jobs are likely
        # to use the same dfi
        delay = (delayed(query_func)(time_binned_dframes[time_idx].copy(), path, crs, *args, **kwargs)
                 for time_idx in time_deps for var, path, crs in time_deps[time_idx])
        point_data_per_time_var = np.array(Parallel(n_jobs=n_jobs, verbose=5)(delay))

        # (time indices * variables, nr of points) -> (time indices, variables, nr of points)
        # print(time_deps)
        # shape = len(time_deps), len(list(time_deps.values())[0]), point_data_per_time_var.shape[1]
        #
        # point_data_per_time_var = np.reshape(point_data_per_time_var, shape)
        
        # iterate over point_data_per_time_var as in Parallel execution above
        running_idx = 0
        for i, time_idx in enumerate(time_deps):
            dfi = time_binned_dframes[time_idx]

            # assign time dependent vars to dframe
            for j, var_at_time in enumerate(time_deps[time_idx]):
                var, path, crs = var_at_time

                if col_name != '':
                    var_name = var + '_' + col_name
                else:
                    var_name = var

                if var not in dframe:
                    dframe[var_name + '_time_delta'] = np.nan
                    dframe[var_name + '_time_ind'] = np.nan
                    dframe[var_name] = np.nan

                dframe.loc[dfi.index, var_name] = point_data_per_time_var[running_idx + j]

                xset_time = xset.coords['time'][time_idx].data
                time_delta = dfi.time.astype(np.datetime64).subtract(xset_time)

                dframe.loc[dfi.index, var_name + '_time_delta'] = time_delta
                dframe.loc[dfi.index, var_name + '_time_ind'] = np.int(time_idx)

            running_idx += len(time_deps[time_idx])

    if time_indeps:
        delay = (delayed(query_func)(dframe.copy(), path, crs, *args, **kwargs)
                 for var, path, crs in time_indeps)
        point_data_per_var = Parallel(n_jobs=n_jobs, verbose=5)(delay)

        # write data to data frame
        for j in range(len(point_data_per_var)):
            var, path, crs = time_indeps[j]

            if col_name != '':
                var_name = var + '_' + col_name
            else:
                var_name = var

            dframe.loc[:, var_name] = point_data_per_var[j]

    return dframe


def _prep_raster_to_point(dframe, xset):
    # for time_mode
    time_binned_dframes = {}

    time_deps = {}
    time_indeps = []
    for var in xset.data_vars:
        if 'time' in xset[var].coords:
            # all vars in xset have same time resolution, so create index and swaths only once
            if time_binned_dframes == {}:
                time_binned_dframes = binarize_dataframe(dframe, 'time', xset[var].coords['time'].data,
                                                         pad_lo=np.datetime64('1970-01-01'),
                                                         pad_hi=np.datetime64('now'))

            for time_idx in time_binned_dframes.keys():
                time = xset[var].coords['time'][time_idx].data.astype(str)[:13]
                path = xset[var].attrs['path'][time]
                crs = xset[var].attrs['crs']

                if time_idx in time_deps:
                    time_deps[time_idx].append((var, path, crs))
                else:
                    time_deps[time_idx] = [(var, path, crs)]

        # if variable is not time dependent
        else:
            time_indeps.append((var, xset[var].attrs['path'], xset[var].attrs['crs']))

    return time_indeps, time_deps, time_binned_dframes


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


def cluster_points(dframe, epsilon, min_samples, *args, **kwargs):
    dframe = to_crs(dframe.copy(), epsg=4326)
    coords = np.radians(dframe[['x', 'y']].values.copy())
    db = DBSCAN(eps=epsilon, min_samples=min_samples, *args, **kwargs)
    clusters = db.fit_predict(coords)
    return clusters, db.core_sample_indices_


