#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:32:26 2019

@author: jim
"""
import pandas as pd
import geopandas as gpd
import os
import h5py
import rasterio as rio
from shapely.geometry import Point, GeometryCollection, MultiPoint
from astropy.time import Time
import glob
import numpy as np
import datetime as dt
import re
from rasterio.mask import mask
from joblib import Parallel, delayed
from . import base_data_structures as bs
from . import helpers as hp

import xarray as xr
import uuid
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from fiona.crs import from_epsg


def rasterio_to_gtiff_xarray(arr, meta, path='.', *args, **kwargs):
    tmp_path = os.path.join(path, '%s.tmp.tif' % str(uuid.uuid4()))
    with rio.open(tmp_path, 'w', **meta) as fil2:
        fil2.write(arr)
    out = xr.open_rasterio(tmp_path, *args, **kwargs)

    out = out.squeeze('band', drop=True)
    out = out.where(out != out.attrs['nodatavals'][0])
    out.attrs['nodatavals'] = np.nan

    os.remove(tmp_path)
    return out

#
# def rasterio_to_gtiff_xarray(arr, meta, path='.', *args, **kwargs):
#     with MemoryFile() as memfile:
#         with memfile.open(**meta) as dataset:
#             dataset.write(arr, 1)
#         out = xr.open_rasterio(memfile, *args, **kwargs)
#     return out


class _Reader(object):
    def __init__(self, path, bbox=None, time=None, *args, **kwargs):
        self.path = path
        self.bbox = bbox
        self.time = time

    def _which_bbox(self, bbox):
        if bbox is None:
            bbox = self.bbox
        return bbox

    def _which_time(self, time):
        if time is None:
            bbox = self.time
        return time

    def query(self, time=None, bbox=None, n_jobs=2, epsg=None, *args, **kwargs):
        pass


class ICESATReader(_Reader):
    atl06 = {'lat': '/land_ice_segments/latitude',
             'lon': '/land_ice_segments/longitude',
             'h': '/land_ice_segments/h_li',
             'h_sigma': '/land_ice_segments/sigma_geo_h',
             'delta_time': '/land_ice_segments/delta_time',
             'q_flag': '/land_ice_segments/atl06_quality_summary',
             't_ref': '/ancillary_data/atlas_sdp_gps_epoch',
             'segment_id': '/land_ice_segments/segment_id',
             'rgt': '/orbit_info/rgt'}

    atl08 = {'lat': '/land_segments/latitude',
             'lon': '/land_segments/longitude',
             'h': '/land_segments/terrain/h_te_best_fit',
             'h_sigma': '/land_segments/terrain/h_te_uncertainty',
             'delta_time': '/land_segments/delta_time',
             'q_flag': '/land_segments/dem_removal_flag',
             't_ref': '/ancillary_data/atlas_sdp_gps_epoch',
             'segment_id': '/land_ice_segments/segment_id',
             'rgt': '/orbit_info/rgt'}

    keywords = {(2, 6): atl06, (2, 8): atl08}

    def __init__(self, dirpath, mission=2, prod_nr=8, bbox=None, time=None):
        _Reader.__init__(self, path=dirpath, bbox=bbox, time=time)
        self.mission = mission
        self.prod_nr = prod_nr
        self.dict = self.keywords[(self.mission, self.prod_nr)]

    def query(self, n_jobs=2, time=None, segments=None, bbox=None, hull=False, alpha=.1, buffer=.1, *args, **kwargs):
        """
        Reads the icesat data files according to the query specifications.
        """
        bbox = self._which_bbox(bbox)
        time = self._which_time(time)

        fnames = glob.glob(os.path.join(self.path, '*.h5'))
        fnames = np.array_split(self.select_files(fnames, time=time, segments=segments), n_jobs)
        dfs, bboxs = zip(*Parallel(n_jobs=n_jobs,
                                   verbose=5)(delayed(self._read)(f, bbox=bbox, *args, **kwargs) for f in fnames))

        dframe, bbox = pd.concat(dfs).reset_index(drop=True), bboxs[0]

        if hull:
            convex_hull, edge_points = hp.concave_hull(dframe.geometry, alpha=alpha)
            epsg = hp.get_epsg_from_geopandas(dframe)
            return dframe, bs.BBox(bbox=convex_hull.buffer(buffer), epsg=epsg, res=bbox.get_resolution(epsg))

        return dframe, bbox

    def _read(self, fnames, bbox=None, quality=0, out=False, values=[], epsg=None, *args, **kwargs):
        """ Params
        ------
            fnames (iter) : iterable of paths
            crs (str) : Coordinate Reference System (as defined by GeoPandas)
            bbox (BBox) : data frame with one polygon entry
            version (int) : which ATL
            quality (int) : use data points with quality flag < quality
            out (bool) : if True, writes data to h5 files, one for every of the six tracks
            values (iter) : path in h5 file under ground track name (e.g. '/land_ice_segments/latitude')

        Returns
        -------
            df (GeoPandas.DataFrame) : GeoPandas.DataFrame with correct crs
        """
        tracks = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
        df = pd.DataFrame()

        # Loop trough files
        for fname in fnames:
            for k, g in enumerate(tracks):

                # -----------------------------------#
                # 1) Read in data for a single beam #
                # -----------------------------------#

                # Load variables into memory (more can be added!)
                custom_vars = {}
                with h5py.File(fname, 'r') as fi:
                    lat = fi[g + self.dict['lat']][:]
                    lon = fi[g + self.dict['lon']][:]
                    h = fi[g + self.dict['h']][:]
                    s_li = fi[g + self.dict['h_sigma']][:]
                    t_dt = fi[g + self.dict['delta_time']][:]
                    q_flag = fi[g + self.dict['q_flag']][:]
                    t_ref = fi[self.dict['t_ref']][:]
                    rgt = fi[self.dict['rgt']][:]

                    for v in values:
                        try:
                            custom_vars[v] = fi[g + self.dict[v]][:]
                        except:
                            print('Could not include ' + v)
                            pass
                # ---------------------------------------------#
                # 2) Filter data according region and quality #
                # ---------------------------------------------#

                # Select a region of interest
                if bbox is not None:
                    lonmin, latmin, lonmax, latmax = bbox.get_bounds(epsg=4326)
                    bbox_mask = (lon >= lonmin) & (lon <= lonmax) & \
                                (lat >= latmin) & (lat <= latmax)
                else:
                    bbox_mask = np.ones_like(lat, dtype=bool)  # get all

                # Only keep good data, and data inside bbox
                mask = (q_flag <= quality) & (np.abs(h) < 10e3) & (bbox_mask == 1)

                # Update variables
                lat, lon, h, s_li, t_dt, q_flag = lat[mask], lon[mask], h[mask], \
                                                  s_li[mask], t_dt[mask], q_flag[mask]

                for v in custom_vars.keys():
                    custom_vars[v] = custom_vars[mask]

                # Test for no data
                if len(h) == 0: continue

                # Test if within time bounds

                # -------------------------------------#
                # 3) Convert time and separate tracks #
                # -------------------------------------#

                # Time in GPS seconds (secs since 1980...)
                t_gps = t_ref + t_dt

                # Time in decimal years
                t_year = self.gps2dyr(t_gps)
                time = np.vectorize(self._decimal_to_datetime)(decimal=t_year)

                t_iso = np.vectorize(lambda d: dt.datetime.strftime(d, format='%Y-%m-%d'))(time)

                # Determine orbit type
                i_asc, i_des = self.track_type(t_year, lat)

                # -----------------------#
                # 4) Save selected data #
                # -----------------------#

                # Save variables
                custom_vars['x'] = lon
                custom_vars['y'] = lat
                custom_vars['height'] = h
                custom_vars['time'] = time
                custom_vars['t_sec'] = t_gps
                custom_vars['time'] = t_iso
                custom_vars['s_elv'] = s_li
                custom_vars['q_flg'] = q_flag
                custom_vars['ascending'] = i_asc
                custom_vars['ground_track_id'] = [g] * len(lon)
                custom_vars['rgt'] = rgt * np.ones(len(lat))

                if out:
                    # Define output file name
                    ofile = fname.replace('.h5', '_' + g + '.h5')
                    fil = h5py.File(ofile, 'w')
                    for v in custom_vars.keys():
                        fil[v] = custom_vars[v]

                    print('out ->', ofile)
                    fil.close()

                f = pd.DataFrame(custom_vars)
                df = pd.concat((df, f), sort=True).reset_index(drop=True)

                # create proper datetime index
                df['time'] = pd.to_datetime(df['time'])

        # create GeoPandas DataFrame for proper registering
        if not df.empty:
            points = [Point(x, y) for x, y in zip(df.x, df.y)]
            df = gpd.GeoDataFrame(df, geometry=points, crs=from_epsg(4326))

            if epsg is not None:
                df = df.to_crs(epsg=epsg)

        return df, bbox

    def _decimal_to_datetime(self, decimal):
        year = np.int(decimal)
        rem = decimal - year

        base = dt.datetime(year, 1, 1)
        return base + dt.timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)

    def segment_from_fname(self, fname):
        """ IS2 fname -> segment number. """
        s = fname.split('_')[2]
        return int(s[-2:])

    def select_files(self, files, segments=None, time=None):
        files_out = []

        if time is None:
            t1, t2 = None, None
        else:
            t1, t2 = time

        for f in files:
            include = True
            fname = os.path.basename(f)
            time = self.time_from_fname(fname)

            if segments is not None:
                segment = self.segment_from_fname(fname)
                include = segment in segments

            if t1 is not None:
                include *= t1 <= time

            if t2 is not None:
                include *= t2 >= time

            if include:
                files_out.append(f)
        return np.array(files_out)

    def gps2dyr(self, time):
        """ Convert GPS time to decimal years. """
        return Time(time, format='gps').decimalyear

    def track_type(self, time, lat):
        """
        Separate tracks into ascending and descending.

        Defines tracks as segments with time breaks > tmax,
        and tests whether lat increases or decreases w/time.
        """
        tracks = np.zeros(lat.shape)  # generate track segment
        tracks[0:np.argmax(np.abs(lat))] = 1  # set values for segment
        i_asc = np.zeros(tracks.shape, dtype=bool)  # output index array

        # Loop trough individual segments
        for track in np.unique(tracks):

            i_track, = np.where(track == tracks)  # get all pts from seg

            if len(i_track) < 2: continue

            # Test if lat increases (asc) or decreases (des) w/time
            i_min = time[i_track].argmin()
            i_max = time[i_track].argmax()
            lat_diff = lat[i_track][i_max] - lat[i_track][i_min]

            # Determine track type
            if lat_diff > 0:  i_asc[i_track] = True

        return i_asc, np.invert(i_asc)  # index vectors

    def time_from_fname(self, fname):
        """ IS2 fname -> datatime object. """
        t = fname.split('_')[1]
        y, m, d, h, mn, s = t[:4], t[4:6], t[6:8], t[8:10], t[10:12], t[12:14]
        time = dt.datetime(int(y), int(m), int(d), int(h), int(mn), int(s))
        return time


class _RasterReader(_Reader):
    """
    _RasterReader is an interface between SWISSMap and rasterio reading. It handles queries related to raster operations
    during reading. Readers specific to some data type or directory structure build on top of _RasterReader.
    """

    def __init__(self, path, bbox=None, time=None, *args, **kwargs):
        _Reader.__init__(self, path, bbox=bbox, time=time)

    def read(self, paths=None, bbox=None, align=True, epsg=4326, *args, **kwargs):
        """
        Read rasters in paths constrained by bbox. No resampling and alignment is performed.
        :param paths:
        :param bbox:
        :param args:
        :param kwargs:
        :return:
        """
        bbox = self._which_bbox(bbox)

        # single file read
        if paths is None:
            paths = self.path
        if type(paths) is str:
            paths = [paths]

        out_xarrs = []
        out_bboxs = []

        if align:
            if bbox is None:
                bbox = bs.BBox.from_tif(paths[0])
            for path in paths:
                # crop tif and save to tmp file
                _, tmp_path = self._crop_tif(path, bbox=bbox, to_file=True)

                # warp image
                out = self._warp_tif(tmp_path, bbox=bbox, epsg=epsg)

                os.remove(tmp_path)

                out.attrs['path'] = path
                out_xarrs.append(out)
                out_bboxs.append(bbox)

        else:
            for path in paths:
                if bbox is not None:
                    out, _ = self._crop_tif(path, bbox=bbox, to_file=False)
                else:
                    with rio.open(path, 'r') as fil:
                        out = xr.open_rasterio(fil, *args, **kwargs)

                    # make bbox that is returned
                    bbox = bs.BBox.from_tif(path)

                out.attrs['path'] = path
                out_xarrs.append(out)
                out_bboxs.append(bbox)

        return out_xarrs, out_bboxs

    def _crop_tif(self, path, bbox, to_file=False):
        with rio.open(path) as fil:
            coords = bbox.get_rasterio_coords(fil.crs.data)
            out_img, out_transform = mask(dataset=fil, shapes=coords, crop=True)
            out_meta = fil.meta.copy()

            out_meta.update({"driver": "GTiff",
                             "height": out_img.shape[1],
                             "width": out_img.shape[2],
                             "transform": out_transform,
                             "count": fil.count,
                             "dtype": out_img.dtype})

        tmp_path = os.path.join('.', '%s.tmp.tif' % str(uuid.uuid4()))
        with rio.open(tmp_path, 'w', **out_meta) as fil:
            fil.write(out_img)

        if to_file:
            return None, tmp_path
        else:
            out = rasterio_to_gtiff_xarray(out_img, out_meta)
            os.remove(tmp_path)
            return out, None

    def _warp_tif(self, path, bbox, epsg, *args, **kwargs):
        left, bottom, right, top = bbox.get_bounds(epsg=epsg)
        res = bbox.get_resolution(epsg)

        width = (right - left) // res[0]
        height = (top - bottom) // res[1]
        dst_transform = rio.transform.from_origin(west=left, north=top, xsize=res[0], ysize=res[1])

        vrt_options = {
            'resampling': Resampling.cubic,
            'crs': CRS.from_epsg(epsg),
            'transform': dst_transform,
            'height': height,
            'width': width,
        }

        with rio.open(path) as src:
            with WarpedVRT(src, **vrt_options) as vrt:

                # At this point 'vrt' is a full dataset with dimensions,
                # CRS, and spatial extent matching 'vrt_options'.
                dta = vrt.read()
                # # Read all data into memory.
                # #
                vrt_meta = vrt.meta.copy()

                vrt_meta.update({"driver": "GTiff",
                                 "height": dta.shape[1],
                                 "width": dta.shape[2],
                                 "transform": dst_transform,
                                 "count": vrt.count})

                xarr = rasterio_to_gtiff_xarray(dta, vrt_meta, *args, **kwargs)

        return xarr


class _TimeRasterReader(_RasterReader):
    """
    _TimeRasterReader handles data sets of rasters with a time label. Time information extraction takes place in
    _create_path_dict.
    """

    def __init__(self, dirpath, bbox=None, time=None, *args, **kwargs):
        _RasterReader.__init__(self, path=dirpath, bbox=bbox, time=time, *args, **kwargs)

        self._path_dict = self._create_path_dict()
        self.min_time = min(self._path_dict.values())
        self.max_time = max(self._path_dict.values())

    def query(self, bbox=None, time=None, align=False, *args, **kwargs):
        bbox = self._which_bbox(bbox)
        time = self._which_time(time)

        if time is None:
            pathes_times = list(self._path_dict.items())
        else:
            start, end = time
            if start is None:
                start = self.min_time
            if end is None:
                end = self.max_time
            pathes_times = list((path, time) for path, time in self._path_dict.items() if start <= time <= end)

        if len(pathes_times) == 0:
            return None, [bbox]

        paths, times = zip(*pathes_times)
        arrs, bboxs = self.read(paths, bbox=bbox, align=True, *args, **kwargs)

        ret = xr.concat(arrs, 'time')
        ret.coords['time'] = ('time', np.array(times))

        return ret, bboxs

    def _create_path_dict(self):
        pass


class SnowCoverReader(_TimeRasterReader):
    def _create_path_dict(self):
        fnames = glob.glob(os.path.join(self.path, '*.tif'))
        pattern = r'([0-9]+)\.tif'

        path_dict = {}
        for fname in fnames:
            date = re.findall(pattern, fname)[-1]
            year = int(date[:4])
            doy = int(date[4:])
            path_dict[os.path.join(self.path, fname)] = dt.datetime(year=year, month=1, day=1) \
                                                        + dt.timedelta(days=doy - 1)

        return path_dict


def read_raster(path, bbox=None, *args, **kwargs):
    return _RasterReader(path, bbox).read(*args, **kwargs)


class SLFReader(_Reader):
    def __init__(self, *args, **kwargs):
        _Reader.__init__(self, *args, **kwargs)
        self._cached_slf_data, self.slf_stations = self.load_slf_data()

    def load_slf_data(self):
        # read station data
        slfstats_file = 'SLF_utm32_cood.csv'
        slfstats = pd.read_csv(os.path.join(self.path, slfstats_file))

        # tidy up station data
        slfstats['slf station code'] = slfstats['slf station code'].map(str) + slfstats['slf_locati,N,10,0'].map(str)
        slfstats.drop(columns=['slf_locati,N,10,0'])

        slf_codes = dict((row['slf station code'],
                          {'geometry':Point(row['X_utm,C,254'],
                                            row['Y_utm,C,254']),
                           'height': row['altitude_a,N,10,0']}) for idx, row in slfstats.iterrows())

        # read time series data
        glob_re = lambda pattern, strings: filter(re.compile(pattern).match, strings)
        fnames = glob_re(r'one_year_imis_\d*\.csv', os.listdir(self.path))

        slf = pd.DataFrame()
        for fname in fnames:
            slf = pd.concat((slf, pd.read_csv(os.path.join(self.path, fname))))

        # tidy up slf
        slf['stat_abk'] = slf['stat_abk'].map(str) + slf['stao_nr'].map(str)
        slf.drop(columns=['stao_nr'])

        slf['time'] = pd.to_datetime(slf['datetime [utc+1]'].map(str), format='%Y%m%d%H%M%S', errors='coerce')
        slf.drop(columns='datetime [utc+1]')
        slf['time'] = pd.to_datetime(slf['time'])

        # convert time series data to gpd.GeoDataFrame by adding location info
        slf['height'] = [slf_codes[code]['height'] if code in slf_codes else np.nan for code in slf['stat_abk']]
        geometry = [slf_codes[code]['geometry'] if code in slf_codes else GeometryCollection() for code in slf['stat_abk']]

        return gpd.GeoDataFrame(slf, crs=from_epsg(4326), geometry=geometry), slf_codes

    def query(self, time=None, bbox=None, epsg=None, *args, **kwargs):
        time = self._which_time(time)
        bbox = self._which_bbox(bbox)

        # take out measurements that are not time frame
        slf = self._filter_time(self._cached_slf_data, time)

        # take out measurements that are not in bbox
        slf = self._filter_bbox(slf, bbox)

        if epsg is not None:
            slf = slf.to_crs(epsg=epsg)

        return slf, bbox

    def _filter_time(self, df, time):
        if time is not None:
            start, end = time
            if start is not None:
                time_idx = df[df['time'] < start].index
                df.drop(time_idx, inplace=True)
            if end is not None:
                time_idx = df[df['time'] > end].index
                df.drop(time_idx, inplace=True)
        return df

    def _filter_bbox(self, df, bbox):
        if bbox is not None:
            df = df.to_crs(bbox.df.crs)
            df = gpd.sjoin(df, bbox.df, how='left')
        return df


if __name__ == '__main__':
    data = '/Volumes/HADDOCK 460GB/swiss_project/data/'

    icesat_path = os.path.join(data, 'icesat2/2019_02_22')
    snowcov_path = os.path.join(data, 'snow_cover/')
    clsf_path = os.path.join(data, 'land_cover/corine/CLC_2012_utm32_DeFROST.tif')
    slf_path = os.path.join(data, 'SLF/one_year_imis_2019.csv')

    ice = ICESATReader(dirpath=icesat_path, mission=2, prod_nr=8)
    gdf = ice.read()
