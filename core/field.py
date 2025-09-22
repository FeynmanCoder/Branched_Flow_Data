# branched_flow/core/field.py

import numpy as np

class FieldData:
    """封装一个二维场及其相关元数据的通用数据结构。"""
    def __init__(self, field_matrix: np.ndarray, x_min: float, y_min: float, dx: float, dy: float):
        self.field_matrix = field_matrix
        self.x_min = x_min
        self.y_min = y_min
        self.dx = dx
        self.dy = dy
        self.ny, self.nx = field_matrix.shape
        self.x_grid = np.linspace(x_min, x_min + (self.nx - 1) * dx, self.nx)
        self.y_grid = np.linspace(y_min, y_min + (self.ny - 1) * dy, self.ny)

    def get_x_grid_coords(self) -> np.ndarray:
        return self.x_grid

    def get_y_grid_coords(self) -> np.ndarray:
        return self.y_grid

    def get_field_matrix(self) -> np.ndarray:
        return self.field_matrix