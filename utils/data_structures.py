import datetime as dt
import xarray as xr
import os
import time as tim

from . import readers as rs


class SWISSMap(rs._Reader):
    def __init__(self, paths, bbox=None, time=None, calc_dir='.', *args, **kwargs):
        """

        :param paths: paths in the following order (sc_path, ice_path, dem_path, slope_path, slf_path, lc_path)
        :param bbox: (bs.BBox)
        :param res: resolution in epsg units
        :param epsg: resolution coordinate unit
        :param time: time period given in isoformat strings (start YYYY-MM-DD, end YYYY-MM-DD)
        """
        rs._Reader.__init__(self, path=paths, bbox=bbox, time=time)

        self.calc_dir = calc_dir

        sc_path, ice_path, dem_path, slope_path, slf_path, lc_path = paths
        self.paths_time_invariant_vars = {'dem': dem_path, 'land_cover': lc_path, 'slope': slope_path}

        # Initialize Readers of time varying variables
        self.snow_cover_reader = rs.SnowCoverReader(sc_path, bbox=bbox, time=time, *args, **kwargs)
        self.icesat_reader = rs.ICESATReader(ice_path, bbox=bbox, time=time, *args, **kwargs)
        self.slf_reader = rs.SLFReader(slf_path, bbox=bbox, time=time, *args, **kwargs)

    def query(self, time=None, bbox=None, segments=None, epsg=3857, query_dir=None, *args, **kwargs):
        time = self._convert_time_to_datetime(time)

        if query_dir is None:
            dirfmt = r'query-%4d-%02d-%02d-%02d-%02d-%02d'
            now = tim.localtime()[0:6]
            query_dir = os.path.join(self.calc_dir, dirfmt % now)
            if not os.path.exists(query_dir):
                os.makedirs(query_dir)

        # Read variables, make sure that, in case no bbox is supplied, data are aligned
        # by overriding bbox
        ice_data, bbox = self.icesat_reader.query(time=time, bbox=bbox, segments=segments,
                                                  epsg=epsg, tmpdir=query_dir, *args, **kwargs)

        snow_data, _ = self.snow_cover_reader.query(time=time, bbox=bbox, epsg=epsg, tmpdir=query_dir, *args, **kwargs)
        qmap, _ = self._get_time_invariant_map(bbox=bbox, epsg=epsg, tmpdir=query_dir, *args, **kwargs)
        # slf_data, _ = self.slf_reader.query(time=time, bbox=bbox, epsg=epsg, tmpdir=query_dir, *args, **kwargs)

        return qmap, snow_data, ice_data, None, bbox

    def _get_time_invariant_map(self, epsg, bbox=None, *args, **kwargs):
        """
        :param res:
        :param epsg:
        :param bbox:
        :return:
        """
        names, paths = list(self.paths_time_invariant_vars.keys()), list(self.paths_time_invariant_vars.values())

        maps, bboxs = rs.read_raster(path=paths, epsg=epsg, bbox=bbox, *args, **kwargs)

        mapping = dict(zip(names, maps))
        return xr.Dataset(mapping), bboxs

    def _convert_time_to_datetime(self, time):
        if time is not None:
            start, end = time
            if type(start) is str:
                start = dt.datetime.fromisoformat(start)
            if type(end) is str:
                end = dt.datetime.fromisoformat(end)
            time = start, end
        return time