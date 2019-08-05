import datetime as dt
import xarray as xr
import os
import time as tim
import pickle as pkl

from . import readers as rs


class SWISSMap(object):
    def __init__(self, slf_path=None, ice_path=None, dem_path=None, slope_path=None, sc_path=None, lc_path=None,
                 bbox=None, time=None, calc_dir='.', load_dir=None, *args, **kwargs):
        """

        :param paths: paths in the following order (sc_path, ice_path, dem_path, slope_path, slf_path, lc_path)
        :param bbox: (bs.BBox)
        :param res: resolution in epsg units
        :param epsg: resolution coordinate unit
        :param time: time period given in isoformat strings (start YYYY-MM-DD, end YYYY-MM-DD)
        """
        self.FIL_NAME_ICE = 'tmp__ice.pkl'
        self.FIL_NAME_SLF = 'tmp__slf.pkl'
        self.FIL_NAME_LOAD_DIR_SPECS = 'query_specs.pkl'
        self.DIR_NAME_SNOW_COVER = 'snow_data'
        self.DIR_NAME_BG_RASTERS = 'bg_rasters_data'
        self.DIR_NAME_TABULAR = 'tabular_data'
        self.FIL_NAMES_TIME_INV_VARS = {'dem': 'tmp__dem.tif', 'land_cover': 'tmp__land_cover.tif',
                                        'slope': 'tmp__slope.tif'}

        self.calc_dir = calc_dir
        self.load_dir = load_dir
        self.load_dir_specs = None

        if load_dir is not None:
            load_dir_specs_path = os.path.join(load_dir, self.FIL_NAME_LOAD_DIR_SPECS)
            assert os.path.exists(load_dir_specs_path)

            with open(load_dir_specs_path, 'rb') as f:
                self.load_dir_specs = pkl.load(f)

            self.paths = self.load_dir_specs['query_paths']

            sc_path, ice_path, slf_path = self.paths['snow'], self.paths['icesat'], self.paths['slf']
            self.paths_bg_raster_vars = self.paths.copy()

            del self.paths_bg_raster_vars['snow']
            del self.paths_bg_raster_vars['icesat']
            del self.paths_bg_raster_vars['slf']

        else:
            assert sc_path is not None and \
                   ice_path is not None and \
                   dem_path is not None and \
                   slope_path is not None and \
                   slf_path is not None and \
                   lc_path is not None

            self.paths_bg_raster_vars, self.paths = self._create_path_dict(dem_path, lc_path, slope_path,
                                                                           sc_path, ice_path, slf_path)

        self.snow_rasters_reader = rs.SnowCoverReader(sc_path, bbox=bbox, time=time, *args, **kwargs)
        self.icesat_reader = rs.ICESATReader(ice_path, bbox=bbox, time=time, *args, **kwargs)
        self.slf_reader = rs.SLFReader(slf_path, bbox=bbox, time=time, *args, **kwargs)

    def query(self, time=None, bbox=None, segments=None, epsg=None, out=False, query_dir_name=None, chunks='auto',
              align=False, *args, **kwargs):
        time = self._convert_time_to_datetime(time)

        is_query_saved = chunks is None or out or align

        dir_names = self._prepare_query_dir(query_dir_name, is_query_saved, time=time, bbox=bbox,
                                            segments=segments, epsg=epsg, chunks=chunks, **kwargs)
        query_dir, load_dir_specs_path, bg_rasters_query_dir, query_dir, snow_query_dir, tabular_query_dir = dir_names

        print('Writing output to', query_dir)

        if chunks == 'auto':
            chunks = {'x':'auto', 'y':'auto'}

        return self._query(time, bbox, segments, epsg, align, chunks,
                           out, is_query_saved, bg_rasters_query_dir,
                           snow_query_dir, tabular_query_dir, *args, **kwargs)

    def _query(self, time, bbox, segments, epsg, align, chunks, out, is_query_saved, bg_rasters_query_dir,
               snow_query_dir, tabular_query_dir, *args, **kwargs):

        # Read variables, make sure that, in case no bbox is supplied, data are aligned
        # by overriding bbox
        ice_data, bbox = self.icesat_reader.query(time=time, bbox=bbox, segments=segments, out=is_query_saved,
                                                  tmp_dir=tabular_query_dir, fil_name=self.FIL_NAME_ICE,
                                                  epsg=epsg, *args, **kwargs)

        snow_rasters, _ = self.snow_rasters_reader.query(time=time, bbox=bbox, epsg=epsg, tmp_dir=snow_query_dir,
                                                         out=out, chunks=chunks, align=align, *args, **kwargs)

        fil_names = list(self.FIL_NAMES_TIME_INV_VARS.values())
        bg_rasters, _ = self._get_bg_rasters(bbox=bbox, epsg=epsg, tmp_dir=bg_rasters_query_dir, out=out,
                                             fil_names=fil_names, chunks=chunks, align=align, *args, **kwargs)

        slf_data, _ = self.slf_reader.query(time=time, bbox=bbox, epsg=epsg, tmp_dir=tabular_query_dir, 
                                            fil_name=self.FIL_NAME_SLF, out=is_query_saved, *args, **kwargs)

        return bg_rasters, snow_rasters, ice_data, slf_data, bbox,

    def _prepare_query_dir(self, query_dir_name, is_query_saved, **kwargs):
        if query_dir_name is None:
            dir_fmt = r'query-%4d-%02d-%02d-%02d-%02d-%02d'
            now = tim.localtime()[0:6]
            query_dir_name = dir_fmt % now

        query_dir = os.path.join(self.calc_dir, query_dir_name)
        snow_query_dir = os.path.join(query_dir, self.DIR_NAME_SNOW_COVER)
        bg_rasters_query_dir = os.path.join(query_dir, self.DIR_NAME_BG_RASTERS)
        tabular_query_dir = os.path.join(query_dir, self.DIR_NAME_TABULAR)
        load_dir_specs_path = os.path.join(query_dir, self.FIL_NAME_LOAD_DIR_SPECS)

        if not os.path.exists(query_dir):
            os.makedirs(query_dir)
            os.makedirs(snow_query_dir)
            os.makedirs(bg_rasters_query_dir)
            os.makedirs(tabular_query_dir)

        if is_query_saved:
            dem_path = os.path.join(bg_rasters_query_dir, self.FIL_NAMES_TIME_INV_VARS['dem'])
            lc_path = os.path.join(bg_rasters_query_dir, self.FIL_NAMES_TIME_INV_VARS['land_cover'])
            slope_path = os.path.join(bg_rasters_query_dir, self.FIL_NAMES_TIME_INV_VARS['slope'])
            ice_path = os.path.join(tabular_query_dir, self.FIL_NAME_ICE)
            slf_path = os.path.join(tabular_query_dir, self.FIL_NAME_SLF)

            _, paths = self._create_path_dict(dem_path, lc_path, slope_path, snow_query_dir, ice_path, slf_path)

        else:
            paths = self.paths

        with open(load_dir_specs_path, 'wb') as f:
            specs = dict(query_paths=paths, is_query_saved_in_query_dir=is_query_saved, query_dir_name=query_dir_name)
            specs.update(kwargs)
            pkl.dump(specs, f)

        return query_dir, load_dir_specs_path, bg_rasters_query_dir, query_dir, snow_query_dir, tabular_query_dir

    def _create_path_dict(self, dem_path, lc_path, slope_path, sc_path, ice_path, slf_path):
        paths_bg_raster_vars = {'dem': dem_path, 'land_cover': lc_path, 'slope': slope_path}
        paths = paths_bg_raster_vars.copy()
        paths.update({'snow': sc_path, 'icesat': ice_path, 'slf': slf_path})

        return paths_bg_raster_vars, paths

    def load(self):
        if self.load_dir_specs['is_query_saved_in_query_dir']:
            self.query(query_dir_name=self.load_dir_specs['query_dir_name'])

        return self.query(**self.load_dir_specs)

    def _get_bg_rasters(self, epsg, bbox=None, *args, **kwargs):
        """
        :param res:
        :param epsg:
        :param bbox:
        :return:
        """
        names, paths = list(self.paths_bg_raster_vars.keys()), list(self.paths_bg_raster_vars.values())
        maps, bboxs = rs.read_raster(path=paths, epsg=epsg, bbox=bbox, *args, **kwargs)

        mapping = dict(zip(names, maps))
        return xr.Dataset(mapping), bboxs

    def _convert_time_to_datetime(self, time):
        if time is not None and type(time[0]) != dt.datetime:
            start, end = time
            if type(start) is str:
                start = dt.datetime.fromisoformat(start)
            if type(end) is str:
                end = dt.datetime.fromisoformat(end)
            time = start, end
        if time is None:
            time = None, None
        return time
