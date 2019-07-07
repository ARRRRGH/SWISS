import datetime as dt
from . import readers as rs
from functools import partial
import xarray as xr

class SWISSMap(object):
    def __init__(self, paths, bbox=None, epsg=3857, time=None, *args, **kwargs):
        """

        :param paths: paths in the following order (sc_path, ice_path, dem_path, slope_path, slf_path, lc_path)
        :param bbox: (bs.BBox)
        :param res: resolution in epsg units
        :param epsg: resolution coordinate unit
        :param time: time period given in isoformat strings (start YYYY-MM-DD, end YYYY-MM-DD)
        """
        sc_path, ice_path, dem_path, slope_path, slf_path, lc_path = self.paths

        # Scalable data
        self.res = res
        self.bbox = bbox

        self.paths = paths
        self.fixed_data_paths = (dem_path, 'dem'), (lc_path, 'land_cover'), (slope_path, 'slope')

        # Dynamic data
        self.snow_cover_reader = rs.SnowCoverReader(sc_path, *args, **kwargs)
        self.icesat_reader = rs.ICESATReader(ice_path, *args, **kwargs)

        # Fixed size data
        self.background_map, self.background_bbox = self._create_bg_map(epsg=epsg, bbox=bbox)
        self.slf_data, self.slf_station_data = rs.read_slf(slf_path)

    @property
    def land_cover(self):
        return self.background_map['land_cover']

    @property
    def dem(self):
        return self.background_map['dem']

    @property
    def slope(self):
        return self.background_map['slope']

    def create_map(self, time=None, bbox=None, segments=None):
        if time is not None:
            start, end = time
        if start is str:
            start = dt.datetime.fromisoformat(start)
        if end is str:
            end = dt.datetime.fromisoformat(end)

        bg = self.background_map.to_array()

        ice_data = self.icesat_reader.query(time=time, bbox=bbox, segments=segments)
        snow_data = self.snow_cover_reader.query(time=time, bbox=bbox)

        bg = xr.concat((bg, snow))


    def _create_bg_map(self, epsg, bbox=None):
        """
        :param res:
        :param epsg:
        :param bbox:
        :return:
        """
        return rs.read_raster(path=self.fixed_data_paths, epsg=epsg, bbox=bbox)




