import sys

sys.path.append("./")

from wp_header import *

import warp as wp
import numpy as np

from amgpcg3d.warp import AMGPCG3DWarp

import time


# Dirichlet BC
# Poisson sin(kx x) * sin(ky y) * sin(kz z)
# [0, 1]  x [0, 1] x [0, 1]
@wp.kernel
def test_c0_init_kernel(
    a_diag: wp.array3d(dtype=wp.float32),
    a_x: wp.array3d(dtype=wp.float32),
    a_y: wp.array3d(dtype=wp.float32),
    a_z: wp.array3d(dtype=wp.float32),
    is_dof: wp.array3d(dtype=wp.uint8),
    b: wp.array3d(dtype=wp.float32),
    dx: float,
    kx: float,
    ky: float,
    kz: float,
):
    i, j, k = wp.tid()
    is_dof[i, j, k] = wp.uint8(1)
    a_diag[i, j, k] = 6.0
    a_x[i, j, k] = -1.0
    a_y[i, j, k] = -1.0
    a_z[i, j, k] = -1.0

    fact = dx * dx * (kx * kx + ky * ky + kz * kz)
    x = (wp.float32(i) + 1.0) * dx
    y = (wp.float32(j) + 1.0) * dx
    z = (wp.float32(k) + 1.0) * dx
    b[i, j, k] = fact * wp.sin(kx * x) * wp.sin(ky * y) * wp.sin(kz * z)


# Neumann BC
# Poisson cos(kx x) * cos(ky y) * cos(kz z)
# [0, 1] x [0, 1] x [0, 1]
@wp.kernel
def test_c1_init_kernel(
    a_diag: wp.array3d(dtype=wp.float32),
    a_x: wp.array3d(dtype=wp.float32),
    a_y: wp.array3d(dtype=wp.float32),
    a_z: wp.array3d(dtype=wp.float32),
    is_dof: wp.array3d(dtype=wp.uint8),
    b: wp.array3d(dtype=wp.float32),
    dx: float,
    kx: float,
    ky: float,
    kz: float,
):
    i, j, k = wp.tid()
    is_dof[i, j, k] = wp.uint8(1)
    diag = 0.0
    a_x[i, j, k] = 0.0
    a_y[i, j, k] = 0.0
    a_z[i, j, k] = 0.0
    if i != 0:
        diag += 1.0
    if i != a_diag.shape[0] - 1:
        diag += 1.0
        a_x[i, j, k] = -1.0
    if j != 0:
        diag += 1.0
    if j != a_diag.shape[1] - 1:
        a_y[i, j, k] = -1.0
        diag += 1.0
    if k != 0:
        diag += 1.0
    if k != a_diag.shape[2] - 1:
        a_z[i, j, k] = -1.0
        diag += 1.0
    a_diag[i, j, k] = diag

    fact = dx * dx * (kx * kx + ky * ky + kz * kz)
    x = (wp.float32(i) + 0.5) * dx
    y = (wp.float32(j) + 0.5) * dx
    z = (wp.float32(k) + 0.5) * dx
    b[i, j, k] = fact * wp.cos(kx * x) * wp.cos(ky * y) * wp.cos(kz * z)


# Hybrid BC
# Laplace (2x - 1)
# [0, 1] x [0, 1] x [0, 1]
@wp.kernel
def test_c2_init_kernel(
    a_diag: wp.array3d(dtype=wp.float32),
    a_x: wp.array3d(dtype=wp.float32),
    a_y: wp.array3d(dtype=wp.float32),
    a_z: wp.array3d(dtype=wp.float32),
    is_dof: wp.array3d(dtype=wp.uint8),
    b: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    is_dof[i, j, k] = wp.uint8(1)
    diag = 2.0
    a_x[i, j, k] = -1.0
    a_y[i, j, k] = 0.0
    a_z[i, j, k] = 0.0
    if j != 0:
        diag += 1.0
    if j != a_diag.shape[1] - 1:
        a_y[i, j, k] = -1.0
        diag += 1.0
    if k != 0:
        diag += 1.0
    if k != a_diag.shape[2] - 1:
        a_z[i, j, k] = -1.0
        diag += 1.0
    a_diag[i, j, k] = diag

    b[i, j, k] = 0.0
    if i == 0:
        b[i, j, k] = -1.0
    elif i == a_diag.shape[0] - 1:
        b[i, j, k] = 1.0


# Dirichlet BC
# uxx+cy uyy + cz uzz = f
# sin(kx x) * sin(ky y) * sin(kz z)
# [0, 1] x [0, 1] x [0, 1]
@wp.kernel
def test_c3_init_kernel(
    a_diag: wp.array3d(dtype=wp.float32),
    a_x: wp.array3d(dtype=wp.float32),
    a_y: wp.array3d(dtype=wp.float32),
    a_z: wp.array3d(dtype=wp.float32),
    is_dof: wp.array3d(dtype=wp.uint8),
    b: wp.array3d(dtype=wp.float32),
    dx: float,
    kx: float,
    ky: float,
    kz: float,
    cy: float,
    cz: float,
):
    i, j, k = wp.tid()
    is_dof[i, j, k] = wp.uint8(1)
    a_diag[i, j, k] = 2.0 + 2.0 * cy + 2.0 * cz
    a_x[i, j, k] = -1.0
    a_y[i, j, k] = -cy
    a_z[i, j, k] = -cz

    fact = dx * dx * (kx * kx + cy * ky * ky + cz * kz * kz)
    x = (wp.float32(i) + 1.0) * dx
    y = (wp.float32(j) + 1.0) * dx
    z = (wp.float32(k) + 1.0) * dx
    b[i, j, k] = fact * wp.sin(kx * x) * wp.sin(ky * y) * wp.sin(kz * z)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=512)
    args = parser.parse_args()

    res = args.res
    res = [res, res, res]

    amgpcg = AMGPCG3DWarp(res, bottom_smoothing=10, verbose=True, trim_info=True)
    amgpcg.clear()

    # case 0
    dx = 1.0 / (res[0] + 1)
    kx = 6 * wp.pi
    ky = 6 * wp.pi
    kz = 6 * wp.pi
    x_grid = np.linspace(dx, 1 - dx, res[0])
    y_grid = np.linspace(dx, 1 - dx, res[1])
    z_grid = np.linspace(dx, 1 - dx, res[2])
    x, y, z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    x_gt = np.sin(kx * x) * np.sin(ky * y) * np.sin(kz * z)

    wp.launch(
        kernel=test_c0_init_kernel,
        dim=res,
        inputs=[
            amgpcg._a_diag,
            amgpcg._a_x,
            amgpcg._a_y,
            amgpcg._a_z,
            amgpcg._is_dof,
            amgpcg._b,
            dx,
            kx,
            ky,
            kz,
        ],
    )
    wp.synchronize()
    t0 = time.perf_counter()
    amgpcg.load_coeff()
    amgpcg.load_rhs()
    amgpcg.build(6.0, -1.0)
    amgpcg.solve(pure_neumann=False)
    wp.synchronize()
    t1 = time.perf_counter()
    print(f"[TEST] case 0: time {(t1 - t0) * 1e3:.2f} ms")

    x_np = amgpcg._x.numpy()
    print(f"[TEST] case 0: L1 {np.mean(np.abs(x_np - x_gt)):.4e}")

    # case 1
    dx = 1.0 / res[0]
    kx = 6 * wp.pi
    ky = 6 * wp.pi
    kz = 6 * wp.pi
    x_grid = np.linspace(dx * 0.5, 1 - dx * 0.5, res[0])
    y_grid = np.linspace(dx * 0.5, 1 - dx * 0.5, res[1])
    z_grid = np.linspace(dx * 0.5, 1 - dx * 0.5, res[2])
    x, y, z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    x_gt = np.cos(kx * x) * np.cos(ky * y) * np.cos(kz * z)

    wp.launch(
        kernel=test_c1_init_kernel,
        dim=res,
        inputs=[
            amgpcg._a_diag,
            amgpcg._a_x,
            amgpcg._a_y,
            amgpcg._a_z,
            amgpcg._is_dof,
            amgpcg._b,
            dx,
            kx,
            ky,
            kz,
        ],
    )
    wp.synchronize()
    t0 = time.perf_counter()
    amgpcg.load_coeff()
    amgpcg.load_rhs()
    amgpcg.build(6.0, -1.0)
    amgpcg.solve(pure_neumann=True)
    wp.synchronize()
    t1 = time.perf_counter()
    print(f"[TEST] case 1: time {(t1 - t0) * 1e3:.2f} ms")

    x_np = amgpcg._x.numpy()
    print(f"[TEST] case 1: L1 {np.mean(np.abs(x_np - x_gt)):.4e}")

    # case 2
    dx = 1.0 / res[0]
    x_grid = np.linspace(dx, 1 - dx, res[0])
    y_grid = np.linspace(dx, 1 - dx, res[1])
    z_grid = np.linspace(dx, 1 - dx, res[2])
    x, y, z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    x_gt = 2 * x - 1

    wp.launch(
        kernel=test_c2_init_kernel,
        dim=res,
        inputs=[
            amgpcg._a_diag,
            amgpcg._a_x,
            amgpcg._a_y,
            amgpcg._a_z,
            amgpcg._is_dof,
            amgpcg._b,
        ],
    )
    wp.synchronize()
    t0 = time.perf_counter()
    amgpcg.load_coeff()
    amgpcg.load_rhs()
    amgpcg.build(6.0, -1.0)
    amgpcg.solve(pure_neumann=False)
    wp.synchronize()
    t1 = time.perf_counter()
    print(f"[TEST] case 2: time {(t1 - t0) * 1e3:.2f} ms")

    x_np = amgpcg._x.numpy()
    print(f"[TEST] case 2: L1 {np.mean(np.abs(x_np - x_gt)):.4e}")

    # case 3
    dx = 1.0 / (res[0] + 1)
    kx = 6 * wp.pi
    ky = 6 * wp.pi
    kz = 6 * wp.pi
    cy = 0.01
    cz = 0.001

    x_grid = np.linspace(dx, 1 - dx, res[0])
    y_grid = np.linspace(dx, 1 - dx, res[1])
    z_grid = np.linspace(dx, 1 - dx, res[2])
    x, y, z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    x_gt = np.sin(kx * x) * np.sin(ky * y) * np.sin(kz * z)

    wp.launch(
        kernel=test_c3_init_kernel,
        dim=res,
        inputs=[
            amgpcg._a_diag,
            amgpcg._a_x,
            amgpcg._a_y,
            amgpcg._a_z,
            amgpcg._is_dof,
            amgpcg._b,
            dx,
            kx,
            ky,
            kz,
            cy,
            cz,
        ],
    )
    wp.synchronize()
    t0 = time.perf_counter()
    amgpcg.load_coeff()
    amgpcg.load_rhs()
    amgpcg.build(2.0 + 2.0 * cy + 2.0 * cz, -1.0)
    amgpcg.solve(pure_neumann=False)
    wp.synchronize()
    t1 = time.perf_counter()
    print(f"[TEST] case 3: time {(t1 - t0) * 1e3:.2f} ms")

    x_np = amgpcg._x.numpy()
    print(f"[TEST] case 3: L1 {np.mean(np.abs(x_np - x_gt)):.4e}")
