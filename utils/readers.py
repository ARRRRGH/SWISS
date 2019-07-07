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
from shapely.geometry import Point
from astropy.time import Time
import glob
import numpy as np
import datetime as dt
import re
from rasterio.mask import mask
from joblib import Parallel, delayed
from . import base_data_structures as bs
import xarray as xr
from rasterio.io import MemoryFile
import uuid
import affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


def rasterio_to_gtiff_xarray(arr, meta, path='.', *args, **kwargs):
    tmp_path = os.path.join(path, '%s.tif' % str(uuid.uuid4()))
    with rio.open(tmp_path, 'w', **meta) as fil2:
        fil2.write(arr)
    out = xr.open_rasterio(tmp_path, *args, **kwargs)
    os.remove(tmp_path)
    return out

#
# def rasterio_to_gtiff_xarray(arr, meta, path='.', *args, **kwargs):
#     with MemoryFile() as memfile:
#         with memfile.open(**meta) as dataset:
#             dataset.write(arr, 1)
#         out = xr.open_rasterio(memfile, *args, **kwargs)
#     return out


class ICESATReader(object):
    atl06 = {'lat': '/land_ice_segments/latitude',
             'lon': '/land_ice_segments/longitude',
             'h': '/land_ice_segments/h_li',
             'h_sigma': '/land_ice_segments/sigma_geo_h',
             'delta_time': '/land_ice_segments/delta_time',
             'q_flag': '/land_ice_segments/atl06_quality_summary',
             't_ref': '/ancillary_data/atlas_sdp_gps_epoch',
             'segment_id': '/land_ice_segments/segment_id'}

    atl08 = {'lat': '/land_segments/latitude',
             'lon': '/land_segments/longitude',
             'h': '/land_segments/terrain/h_te_best_fit',
             'h_sigma': '/land_segments/terrain/h_te_uncertainty',
             'delta_time': '/land_segments/delta_time',
             'q_flag': '/land_segments/dem_removal_flag',
             't_ref': '/ancillary_data/atlas_sdp_gps_epoch',
             'segment_id': '/land_ice_segments/segment_id'}

    keywords = {(2, 6): atl06, (2, 8): atl08}

    def __init__(self, dirpath, mission=2, prod_nr=6):
        self.dirpath = dirpath
        self.mission = mission
        self.prod_nr = prod_nr
        self.dict = self.keywords[(self.mission, self.prod_nr)]

    def query(self, n_jobs=2, time=None, segments=None, *args, **kwargs):
        """
        Reads the icesat data files according to the query specifications.
        """
        fnames = glob.glob(os.path.join(self.dirpath, '*.h5'))
        fnames = np.array_split(self.select_files(fnames, time=time, segments=segments), n_jobs)
        return pd.concat(Parallel(n_jobs=n_jobs, verbose=5)(delayed(self._query)(f, *args, **kwargs) for f in fnames))

    def _query(self, fnames, bbox=None, quality=0, out=False, values=[], *args, **kwargs):
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

                # Time in GPS seconds (secs sinde 1980...)
                t_gps = t_ref + t_dt

                # Time in decimal years
                t_year = self.gps2dyr(t_gps)

                # Determine orbit type
                i_asc, i_des = self.track_type(t_year, lat)

                # -----------------------#
                # 4) Save selected data #
                # -----------------------#

                # Save variables
                custom_vars['lon'] = lon
                custom_vars['lat'] = lat
                custom_vars['h_elv'] = h
                custom_vars['t_year'] = t_year
                custom_vars['t_sec'] = t_gps
                custom_vars['s_elv'] = s_li
                custom_vars['q_flg'] = q_flag
                custom_vars['ascending'] = i_asc
                custom_vars['ground_track_id'] = [g] * len(lon)

                if out:
                    # Define output file name
                    ofile = fname.replace('.h5', '_' + g + '.h5')
                    fil = h5py.File(ofile, 'w')
                    for v in custom_vars.keys():
                        fil[v] = custom_vars[v]

                    print('out ->', ofile)
                    fil.close()

                f = pd.DataFrame(custom_vars)
                df = pd.concat((df, f), sort=True)

        # create GeoPandas DataFrame for proper registering
        if not df.empty:
            points = [Point(x, y) for x, y in zip(df.lon, df.lat)]
            df = gpd.GeoDataFrame(df, geometry=points, crs='epsg:4326')
        return df

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

    def track_type(self, time, lat, tmax=1):
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


class _RasterReader(object):
    """
    _RasterReader is an interface between SWISSMap and rasterio reading. It handles queries related to raster operations
    during reading. Readers specific to some data type or directory structure build on top of _RasterReader.
    """

    def __init__(self, path, bbox=None):
        self.path = path
        self.bbox = bbox

        if self.bbox is not None:
            assert type(self.bbox) is bs.BBox

    def read(self, path=None, bbox=None, align=True, *args, **kwargs):
        if path is None:
            path = self.path

        # single read
        if type(path) is str:
            return self._read(path, bbox=bbox, *args, **kwargs)

        # multiple read
        elif hasattr(path, '__iter__') and not align:
            return [self._read(p, bbox=bbox, *args, **kwargs) for p in path]

        # read and align
        elif hasattr(path, '__iter__') and align:
            return self._read_and_align(path, bbox=bbox, *args, **kwargs)

        # raise exception
        else:
            raise ValueError('path must be str or an iterable of str \
                              or must be supplied during initialization')

    def _read(self, path, bbox=None, *args, **kwargs):
        """
        Wrapper for rasterio's read. allows to specify a bounding box. args and kwargs according to rasterio's read
        or, when bbox is specified, according to rasterio.mask.mask
        :param path:
        :param bbox:
        :param args:
        :param kwargs:
        :return:
        """
        bbox = self._which_bbox(bbox)

        if bbox is not None:
            with rio.open(path) as fil:
                coords = bbox.get_rasterio_coords(fil.crs.data)
                out_img, out_transform = mask(dataset=fil, shapes=coords,
                                              crop=True, *args, **kwargs)
                out_meta = fil.meta.copy()

                out_meta.update({"driver": "GTiff",
                                 "height": out_img.shape[1],
                                 "width": out_img.shape[2],
                                 "transform": out_transform,
                                 "count": fil.count,
                                 "dtype": out_img.dtype})

            out = rasterio_to_gtiff_xarray(out_img, out_meta, *args, **kwargs)
        else:
            with rio.open(path) as fil:
                out = xr.open_rasterio(fil, *args, **kwargs)

        out.attrs['path'] = path
        return out

    def _read_and_align(self, paths, bbox=None, epsg=3857, *args, **kwargs):
        bbox = self._which_bbox(bbox)

        paths, path_names = zip(*paths)

        if bbox is not None:
            left, bottom, right, top = bbox.get_bounds(epsg=epsg)
            res = bbox.get_resolution(epsg)
        else:
            with rio.open(paths[0]) as fil:
                bbox = fil.bounds
                fil_epsg = rio.crs.CRS.to_epsg(fil.crs)

                bbox = bs.BBox.from_rasterio_bbox(bbox, fil_epsg)
                left, bottom, right, top = bbox.get_bounds(epsg)

                res = fil.res
                bbox.set_resolution(res, fil_epsg)

        height = (right - left) // res[0]
        width = (top - bottom) // res[1]
        dst_transform = affine.Affine(res[0], 0.0, left, 0.0, -res[1], top)

        vrt_options = {
            'resampling': Resampling.cubic,
            'crs': CRS.from_epsg(epsg),
            'transform': dst_transform,
            'height': height,
            'width': width,
        }

        out = None
        for path, path_name in zip(paths, path_names):
            with rio.open(path) as src:
                with WarpedVRT(src, **vrt_options) as vrt:

                    # At this point 'vrt' is a full dataset with dimensions,
                    # CRS, and spatial extent matching 'vrt_options'.

                    # Read all data into memory.
                    dta = vrt.read()

                    # Process the dataset in chunks.  Likely not very efficient.
                    # for _, window in vrt.block_windows():
                    #     dta = vrt.read(window=window)

                    vrt_meta = vrt.meta.copy()

                    vrt_meta.update({"driver": "GTiff",
                                     "height": dta.shape[1],
                                     "width": dta.shape[2],
                                     "transform": vrt_options['transform'],
                                     "count": vrt.count})

                    if out is None:
                        out = rasterio_to_gtiff_xarray(dta, vrt_meta, *args, **kwargs).to_dataset(name=path_name)
                    else:
                        out[path_name] = rasterio_to_gtiff_xarray(dta, vrt_meta, *args, **kwargs)

        return out, bbox

    def _which_bbox(self, bbox):
        if bbox is None:
            bbox = self.bbox
        return bbox


class _TimeRasterReader(_RasterReader):
    """
    _TimeRasterReader handles data sets of rasters with a time label. Time information extraction takes place in
    _create_path_dict.
    """

    def __init__(self, dirpath, bbox=None, time=None):
        _RasterReader.__init__(self, dirpath, bbox)
        self.time = time

        self._path_dict = self._create_path_dict()
        self.min_time = min(self._path_dict.values())
        self.max_time = max(self._path_dict.values())

    def query(self, bbox=None, time=None, *args, **kwargs):
        bbox = self._which_bbox(bbox)

        if time is None:
            pathes_times = list(self._path_dict)
        else:
            start, end = time
            if start is None:
                start = self.min_time
            if end is None:
                end = self.max_time
            pathes_times = list((path, time) for path, time in self._path_dict.items() if start <= time <= end)

        ret = None
        for path, time in pathes_times:
            arr = self.read(path, bbox=bbox, *args, **kwargs).expand_dims('time')
            arr.coords['time'] = ('time', [time])

            if ret is None:
                ret = arr
            else:
                ret = xr.concat((ret, arr), 'time')
        return ret

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


def read_slf(path):
    # read tmp data
    glob_re = lambda pattern, strings: filter(re.compile(pattern).match, strings)
    fnames = glob_re(r'one_year_imis_\d*\.csv', os.listdir(path))

    slf = pd.DataFrame()
    for fname in fnames:
        slf = pd.concat((slf, pd.read_csv(os.path.join(path, fname))))

    # read station data
    slfstats_file = 'SLF_utm32_cood.csv'
    slfstats = pd.read_csv(os.path.join(path, slfstats_file))

    return slf, slfstats


if __name__ == '__main__':
    data = '/Volumes/HADDOCK 460GB/swiss_project/data/'

    icesat_path = os.path.join(data, 'icesat2/2019_02_22')
    snowcov_path = os.path.join(data, 'snow_cover/')
    clsf_path = os.path.join(data, 'land_cover/corine/CLC_2012_utm32_DeFROST.tif')
    slf_path = os.path.join(data, 'SLF/one_year_imis_2019.csv')

    ice = ICESATReader(dirpath=icesat_path, mission=2, prod_nr=8)
    gdf = ice.read()
