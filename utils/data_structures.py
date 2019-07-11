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

        # make sure that, in case no bbox is supplied, data are aligned with qmap
        bbox = bboxs[0]

        # Read time dependent variables
        snow_data, bboxes = self.snow_cover_reader.query(time=time, bbox=bbox, align=False, epsg=epsg, *args, **kwargs)

        ice_data, _ = self.icesat_reader.query(time=time, bbox=bbox, segments=segments, epsg=epsg, *args, **kwargs)
        slf_data, _ = self.slf_reader.query(time=time, bbox=bbox, epsg=epsg, *args, **kwargs)

        return qmap, snow_data, ice_data, slf_data

    @staticmethod
    def raster_to_point(dframe, xset, method='nearest', radius_of_influence=1000, inplace=True, *args, **kwargs):
        resample_funcs = {'nearest': pyr.kd_tree.resample_nearest}
        resample_func = resample_funcs[method]

        grid = None
        swath = None
        _time_binned_swaths = {}
        _time_binned_dframes = {}

        # make sure we are in lat / lon coordinates which is what pyresample assumes
        assert dframe.crs['init'] == 'epsg:4326'

        # make sure we are in lat / lon coordinates which pyresample assumes
        # since all vars in xset are aligned, check only first one
        assert list(xset.items())[0][1].attrs['crs'] == '+init=epsg:4326'

        for var in xset.data_vars:
            # all vars in xset are aligned
            if grid is None:
                lats = xset[var]['y']
                lons = xset[var]['x']
                lons, lats = np.meshgrid(lons, lats)
                grid = pyr.geometry.GridDefinition(lats=lats, lons=lons)

            if 'time' in xset[var].coords:
                # all vars in xset have same time resolution, so create index and swaths only once
                if _time_binned_swaths == {}:
                    ts = xset[var].coords['time'].data.astype(dt.datetime)

                    bins = pd.IntervalIndex.from_tuples([((t0 + t1) / 2, (t1 + t2) / 2)
                                                         for t0, t1, t2 in zip(ts, ts[1:], ts[2:])])
                    idxs = pd.cut(dframe['time'], bins=bins, labels=False)

                    for i in np.unique(idxs):
                        dfi = dframe.iloc[np.where(idxs == i)[0]]
                        _time_binned_swaths[i] = pyr.geometry.SwathDefinition(dfi.y, dfi.x)
                        _time_binned_dframes[i] = dfi

                times = [zip(xset[var].isel(time=i),
                             _time_binned_dframes[i],
                             _time_binned_swaths[i])
                         for i in _time_binned_swaths.keys()]

            else:
                # same data points for all vars
                if swath is None:
                    swath = pyr.geometry.SwathDefinition(lats=dframe.y, lons=dframe.x)

                times = [(xset[var], dframe, swath)]

            if not inplace:
                dframe = dframe.copy()

            dframe[var] = np.nan
            for var_at_t, dframe_at_t, swath_at_t in times:
                data = resample_func(source_geo_def=grid,
                                     target_geo_def=swath_at_t,
                                     data=var_at_t.data,
                                     radius_of_influence=radius_of_influence,
                                     *args, **kwargs)

                dframe.loc[dframe_at_t.index, var] = pd.Series(data)

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




