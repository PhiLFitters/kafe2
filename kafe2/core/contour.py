import numpy as np
class ContourException(Exception):
    pass

class ContourFactory():
    
    @staticmethod
    def create_xy_contour(xy_points, sigma):
        xy_points = np.asarray(xy_points)
        _shape = xy_points.shape
        if len(_shape) != 2 or (_shape[0] != 2 and _shape[1] != 2):
            raise ContourException("Explicit contours can only be created from iterables with shape (2,n) or (n,2).")
        if _shape[0] != 2:
            xy_points = xy_points.T
        return Contour(xy_points=xy_points, sigma=sigma)
    
    @staticmethod
    def create_grid_contour(grid_x, grid_y, grid_z, sigma):
        grid_x = np.asarray(grid_x)
        grid_y = np.asarray(grid_y)
        grid_z = np.asarray(grid_z)
        _shape_x = grid_x.shape
        _shape_y = grid_y.shape
        _shape_z = grid_z.shape
        if len(_shape_x) != 1:
            raise ContourException("grid_x needs to be one-dimensional.")
        if len(_shape_y) != 1:
            raise ContourException("grid_y needs to be one-dimensional.")
        if len(_shape_z) != 2:
            raise ContourException("grid_z needs to be two-dimensional.")
        if _shape_x[0] != _shape_z[0]:
            raise ContourException("grid_z needs to be as wide as grid_x is long.")
        if _shape_y[0] != _shape_z[1]:
            raise ContourException("grid_z needs to be as high as grid_y is long.")
        return Contour(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, sigma=sigma)
    

class Contour(object):
    
    def __init__(self, xy_points=None, grid_x=None, grid_y=None, grid_z=None, sigma=None):
        if sigma is None:
            raise ContourException("sigma must not be None.")
        self._xy_points = xy_points
        self._grid_x = grid_x
        self._grid_y = grid_y
        self._grid_z = grid_z
        self._sigma = sigma

    @property
    def xy_points(self):
        return self._xy_points
    
    @property
    def grid_x(self):
        return self._grid_x
    
    @property
    def grid_y(self):
        return self._grid_y
    
    @property
    def grid_z(self):
        return self._grid_z
    
    @property
    def sigma(self):
        return self._sigma
    
