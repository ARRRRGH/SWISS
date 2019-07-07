import geopandas as gpd
from fiona.crs import from_epsg
from shapely.geometry import box, Point
import json

class BBox(object):
    def __init__(self, bbox, epsg=4326, res=None):
        assert type(bbox) == tuple

        self._epsg = epsg
        bbox = box(*bbox)
        self.df = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(epsg))

        if res is None:
            self._res = 1, 1

    @property
    def epsg(self):
        return self._epsg

    def set_resolution(self, res, epsg):
        self._epsg = epsg
        self.df = self.df.to_crs(epsg=epsg)
        self._res = res

    def get_resolution(self, epsg=None):
        if epsg is None:
            return self._res
        else:
            return self._project_resolution(epsg)

    def _project_resolution(self, epsg):
        left, bottom, right, top = self.get_bounds()

        mid_point_coords = (left + right) // 2, (top + bottom) // 2
        print(mid_point_coords, self._res)
        ref_point_coords = mid_point_coords[0] + self._res[0], mid_point_coords[1] + self._res[1]
        pts = [Point(coords) for coords in (mid_point_coords, ref_point_coords)]

        df = gpd.GeoDataFrame({'geometry': pts}, crs=from_epsg(self.epsg))
        df = df.to_crs(epsg=epsg)
        pts = df['geometry']

        return abs(pts[1].coords[0][0] - pts[0].coords[0][0]), abs(pts[1].coords[0][1] - pts[0].coords[0][1])

    def _get(self, epsg=None):
        if epsg is not None:
            return self.df.to_crs(epsg=epsg)
        return self.df

    def get_bounds(self, epsg=None):
        return self._get(epsg)['geometry'][0].bounds

    def get_rasterio_coords(self, crs=None):
        # https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
        if crs is not None:
            df = self.df.to_crs(crs=crs)
        else:
            df = self.df
        return [json.loads(df.to_json())['features'][0]['geometry']]

    @staticmethod
    def from_rasterio_bbox(bbox, epsg):
        return BBox((bbox.left, bbox.bottom, bbox.right, bbox.top), epsg)
