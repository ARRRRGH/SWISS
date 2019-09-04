import geopandas as gpd
from fiona.crs import from_epsg
from shapely.geometry import box, Point, mapping, MultiPolygon
import rasterio as rio
import shapely as shp


class BBox(object):
    def __init__(self, bbox, epsg=4326, res=None):
        self._epsg = epsg
        if type(bbox) == tuple and len(bbox) == 4:
            shape = [box(*bbox)]
        elif type(bbox) == shp.geometry.polygon.Polygon:
            shape = [bbox]
        elif type(bbox) == shp.geometry.MultiPolygon:
            shape = [p for p in bbox]
        else:
            raise ValueError('bbox must be a quadruple or a shapely.geometry.polygon '
                             'or shapely.geometry.MultiplePolygon')

        self.df = gpd.GeoDataFrame({'geometry': shape}, crs=from_epsg(epsg))
        self.df = self.df.reset_index(drop=True)

        if res is None:
            self._res = 1, 1
        else:
            self._res = res

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
        refx_point_coords = mid_point_coords[0] + self._res[0], mid_point_coords[1]
        refy_point_coords = mid_point_coords[0], mid_point_coords[1]  + self._res[1]
        pts = [Point(coords) for coords in (mid_point_coords, refx_point_coords, refy_point_coords)]

        df = gpd.GeoDataFrame({'geometry': pts}, crs=from_epsg(self.epsg))
        df = df.to_crs(epsg=epsg)
        pts = df['geometry']

        return abs(pts[1].coords[0][0] - pts[0].coords[0][0]), abs(pts[2].coords[0][1] - pts[0].coords[0][1])

    def _get(self, epsg=None):
        if epsg is not None:
            return self.df.to_crs(epsg=epsg)
        return self.df

    def get_bounds(self, epsg=None):
        return MultiPolygon([p for p in self._get(epsg=epsg)['geometry']]).bounds

    def get_rasterio_coords(self, crs=None):
        # https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
        if crs is not None:
            df = self.df.to_crs(crs=crs)
        else:
            df = self.df
        # return [f['geometry'] for f in json.loads(df.to_json())['features']]
        return [mapping(geom) for geom in df['geometry']]

    def to_xlim(self, epsg=None):
        bounds = self.get_bounds(epsg)
        return bounds[0], bounds[2]

    def to_ylim(self, epsg=None):
        bounds = self.get_bounds(epsg)
        return bounds[1], bounds[3]

    @staticmethod
    def from_rasterio_bbox(bbox, epsg):
        return BBox((bbox.left, bbox.bottom, bbox.right, bbox.top), epsg)

    @staticmethod
    def from_tif(path):
        with rio.open(path) as fil:
            bbox = fil.bounds
            fil_epsg = rio.crs.CRS.to_epsg(fil.crs)

            bbox = BBox.from_rasterio_bbox(bbox, fil_epsg)

            res = fil.res
            bbox.set_resolution(res, fil_epsg)
        return bbox
