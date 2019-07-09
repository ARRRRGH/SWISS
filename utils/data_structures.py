import datetime as dt
from . import readers as rs
import xarray as xr
import rasterio as rio
import numpy as np
import pyresample as pyr
import pandas as pd
from scipy.spatial import cKDTree

class SWISSMap(rs._Reader):
    def __init__(self, paths, bbox=None, time=None, *args, **kwargs):
        """

        :param paths: paths in the following order (sc_path, ice_path, dem_path, slope_path, slf_path, lc_path)
        :param bbox: (bs.BBox)
        :param res: resolution in epsg units
        :param epsg: resolution coordinate unit
        :param time: time period given in isoformat strings (start YYYY-MM-DD, end YYYY-MM-DD)
        """
        rs._Reader.__init__(self, path=paths, bbox=bbox, time=time)

        sc_path, ice_path, dem_path, slope_path, slf_path, lc_path = paths
        self.paths_time_invariant_vars = {'dem': dem_path, 'land_cover': lc_path, 'slope': slope_path}

        # Initialize Readers of time varying variables
        self.snow_cover_reader = rs.SnowCoverReader(sc_path, bbox=bbox, time=time, *args, **kwargs)
        self.icesat_reader = rs.ICESATReader(ice_path, bbox=bbox, time=time, *args, **kwargs)
        self.slf_reader = rs.SLFReader(slf_path, bbox=bbox, time=time, *args, **kwargs)

    def query(self, time=None, bbox=None, segments=None, epsg=3857, *args, **kwargs):
        if time is not None:
            start, end = time
            if type(start) is str:
                start = dt.datetime.fromisoformat(start)
            if type(end) is str:
                end = dt.datetime.fromisoformat(end)
            time = start, end

        # Read time invariant variables
        qmap, bboxs = self._get_time_invariant_map(bbox=bbox, epsg=epsg, *args, **kwargs)

        # clean qmap
        qmap = qmap.squeeze('band')
        for var in qmap.variables:
            qmap[var] = qmap[var].where(qmap[var] != qmap[var].attrs['nodatavals'][0])

        # make sure that, in case no bbox is supplied, data are aligned with qmap
        bbox = bboxs[0]

        # Read time dependent variables
        snow_data, bboxes = self.snow_cover_reader.query(time=time, bbox=bbox, align=False, epsg=epsg, *args, **kwargs)

        ice_data, _ = self.icesat_reader.query(time=time, bbox=bbox, segments=segments, epsg=epsg, *args, **kwargs)
        slf_data, _ = self.slf_reader.query(time=time, bbox=bbox, epsg=epsg, *args, **kwargs)

        return qmap, snow_data, ice_data, slf_data

    def raster_to_point(self, dframe, xset, method='nearest', *args, **kwargs):
        resample_funcs = {'nearest': pyr.kd_tree.resample_nearest}
        resample_func = resample_funcs[method]

        pts = pyr.geometry.SwathDefinition(dframe.y, dframe.x)
        _time_binned_pts = {}

        for var in xset.variables:
            if 'time' in xset[var].coords:
                if _time_binned_pts == {}:
                    ts = xset[var].coords['time']
                    bins = pd.IntervalIndex.from_tuples([(x, y) for x, y in zip(ts[::2], ts[1::2])])
                    idxs = pd.cut(dframe['time'], bins=bins, labels=False)

                    for i in np.unique(idxs):
                        _time_binned_pts[i] = dframe.iloc[idxs]

                times = zip(xset.isel(time=i)[var], _time_binned_pts)

            else:
                times = (xset[var], dframe)

            # make sure we are in correct projection
            assert xset[var].attrs['crs'] == '+init=epsg:4326'

            dframe[var] = np.nan
            for grid, pts in times:
                griddef = pyr.geometry.GridDefinition(lats=grid['y'], lons=grid['x'])
                data = resample_func(source_geo_def=griddef,
                                     target_geo_def=pts,
                                     data=griddef,
                                     radius_of_influence=50000)
                dframe.index[pts.index][var] = data

        return dframe




    def _get_time_invariant_map(self, epsg, bbox=None, *args, **kwargs):
        """
        :param res:
        :param epsg:
        :param bbox:
        :return:
        """
        names, paths = list(self.paths_time_invariant_vars.keys()), list(self.paths_time_invariant_vars.values())

        maps, bboxs = rs.read_raster(path=paths, epsg=epsg, bbox=bbox,
                                     align=True, *args, **kwargs)
        mapping = dict(zip(names, maps))

        return xr.Dataset(mapping), bboxs




