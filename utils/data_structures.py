import datetime as dt
import xarray as xr

from . import readers as rs
from . import helpers as hp


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
        time = self._convert_time_to_datetime(time)

        # Read variables, make sure that, in case no bbox is supplied, data are aligned
        # by overriding bbox
        ice_data, bbox = self.icesat_reader.query(time=time, bbox=bbox, segments=segments, epsg=epsg, *args, **kwargs)

        qmap, _ = self._get_time_invariant_map(bbox=bbox, epsg=epsg, *args, **kwargs)
        snow_data, _ = self.snow_cover_reader.query(time=time, bbox=bbox, align=True, epsg=epsg, *args, **kwargs)
        slf_data, _ = self.slf_reader.query(time=time, bbox=bbox, epsg=epsg, *args, **kwargs)

        return qmap, snow_data, ice_data, slf_data, bbox

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

        # since align=True all bboxs are equal
        bbox = bboxs[0]

        mapping = dict(zip(names, maps))
        return xr.Dataset(mapping), bbox

    def _convert_time_to_datetime(self, time):
        if time is not None:
            start, end = time
            if type(start) is str:
                start = dt.datetime.fromisoformat(start)
            if type(end) is str:
                end = dt.datetime.fromisoformat(end)
            time = start, end
        return time

    def query_table(self, time=None, bbox=None, segments=None, epsg=3857, *args, **kwargs):
        ice_data, bbox = self.icesat_reader.query(time=time, bbox=bbox, segments=segments, epsg=epsg, *args, **kwargs)
        _ = hp.raster_to_point(ice_data, qmap)
