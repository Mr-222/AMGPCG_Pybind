import torch
import warp as wp

from amgpcg_torch import AMGPCGTorch


class AMGPCGWarp:
    def __init__(
        self,
        raw_res,
        bottom_smoothing: int = 10,
        verbose: bool = False,
        iter_info: bool = False,
        trim_info: bool = False,
    ):
        self._bottom_smoothing = bottom_smoothing
        self._verbose = verbose
        self._tile_size = 8
        tile_dim = [
            (raw_res[0] + self._tile_size - 1) // self._tile_size,
            (raw_res[1] + self._tile_size - 1) // self._tile_size,
            (raw_res[2] + self._tile_size - 1) // self._tile_size,
        ]
        self._res = [
            tile_dim[0] * self._tile_size,
            tile_dim[1] * self._tile_size,
            tile_dim[2] * self._tile_size,
        ]
        self._tile_dim = tile_dim

        self._amgpcgtorch = AMGPCGTorch(
            self._tile_dim, self._bottom_smoothing, self._verbose, iter_info, trim_info
        )

        self._a_diag = wp.zeros(self._res, wp.float32)
        self._a_x = wp.zeros(self._res, wp.float32)
        self._a_y = wp.zeros(self._res, wp.float32)
        self._a_z = wp.zeros(self._res, wp.float32)
        self._is_dof = wp.zeros(self._res, wp.uint8)
        self._b = wp.zeros(self._res, wp.float32)
        self._x = wp.zeros(self._res, wp.float32)

    def clear(self):
        self._a_diag.zero_()
        self._a_x.zero_()
        self._a_y.zero_()
        self._a_z.zero_()
        self._is_dof.zero_()
        self._b.zero_()
        self._x.zero_()

    def load_coeff(self):
        self._amgpcgtorch.load_coeff(
            wp.to_torch(self._a_diag),
            wp.to_torch(self._a_x),
            wp.to_torch(self._a_y),
            wp.to_torch(self._a_z),
            wp.to_torch(self._is_dof),
        )

    def extract_coeff(self):
        self._amgpcgtorch.extract_coeff(
            wp.to_torch(self._a_diag),
            wp.to_torch(self._a_x),
            wp.to_torch(self._a_y),
            wp.to_torch(self._a_z),
            wp.to_torch(self._is_dof),
        )

    def load_rhs(self):
        self._amgpcgtorch.load_rhs(wp.to_torch(self._b))

    def extract_rhs(self):
        self._amgpcgtorch.extract_rhs(wp.to_torch(self._b))

    def build(self, default_a_diag: float, default_a_off_diag: float):
        self._amgpcgtorch.build(default_a_diag, default_a_off_diag)

    def solve(
        self,
        pure_neumann: bool = False,
        rel_tol: float = 1e-12,
        abs_tol: float = 1e-14,
        max_iter: int = 400,
    ):
        self._amgpcgtorch.solve(
            wp.to_torch(self._x),
            pure_neumann,
            rel_tol,
            abs_tol,
            max_iter,
        )
