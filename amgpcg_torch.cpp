#include "solver/amgpcg.h"
#include <torch/extension.h>

#include "data_io.h"
#include <cstdio>

using namespace lfm;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

class AMGPCGTorch {
public:
    AMGPCGTorch(std::vector<int> _tile_dim_vec,
        int _bottom_smoothing = 10, bool _verbose = false, bool _iter_info = false,
        bool _trim_info = false)
        : verbose_(_verbose)
    {
        int3 tile_dim = make_int3(_tile_dim_vec[0], _tile_dim_vec[1], _tile_dim_vec[2]);
        tile_dim_ = tile_dim;

        int level_num = 3;
        // int3 tile_dim_tmp = tile_dim;
        // while (true) {
        //     level_num++;
        //     if (tile_dim_tmp.x % 2 == 0) {
        //         tile_dim_tmp.x /= 2;
        //     } else {
        //         break;
        //     }
        //     if (tile_dim_tmp.y % 2 == 0) {
        //         tile_dim_tmp.y /= 2;
        //     } else {
        //         break;
        //     }
        //     if (tile_dim_tmp.z % 2 == 0) {
        //         tile_dim_tmp.z /= 2;
        //     } else {
        //         break;
        //     }
        // }

        amgpcg_.Alloc(tile_dim, level_num);
        amgpcg_.bottom_smoothing_ = _bottom_smoothing;
        amgpcg_.verbose_ = _verbose;
        amgpcg_.iter_info_ = _iter_info;
        amgpcg_.trim_info_ = _trim_info;
        if (verbose_) {
            printf("[AMGPCG] Solver Constructed\n");
        }
    }
    ~AMGPCGTorch()
    {
    if (verbose_) {
        printf("[AMGPCG] Solver Destructed\n");
        }
    }
    void load_coeff(torch::Tensor _a_diag, torch::Tensor _a_x,
        torch::Tensor _a_y, torch::Tensor _a_z, torch::Tensor _is_dof)
    {
        CHECK_INPUT(_a_diag);
        CHECK_INPUT(_a_x);
        CHECK_INPUT(_a_y);
        CHECK_INPUT(_a_z);
        CHECK_INPUT(_is_dof);

        ConToTileAsync<float>(*(amgpcg_.poisson_vector_[0].a_diag_), tile_dim_, _a_diag, sim_stream);
        ConToTileAsync<float>(*(amgpcg_.poisson_vector_[0].a_x_), tile_dim_, _a_x, sim_stream);
        ConToTileAsync<float>(*(amgpcg_.poisson_vector_[0].a_y_), tile_dim_, _a_y, sim_stream);
        ConToTileAsync<float>(*(amgpcg_.poisson_vector_[0].a_z_), tile_dim_, _a_z, sim_stream);
        ConToTileAsync<uint8_t>(*(amgpcg_.poisson_vector_[0].is_dof_), tile_dim_, _is_dof, sim_stream);
        cudaStreamSynchronize(sim_stream);
    }
    void extract_coeff(torch::Tensor _a_diag, torch::Tensor _a_x,
        torch::Tensor _a_y, torch::Tensor _a_z, torch::Tensor _is_dof)
    {
        CHECK_INPUT(_a_diag);
        CHECK_INPUT(_a_x);
        CHECK_INPUT(_a_y);
        CHECK_INPUT(_a_z);
        CHECK_INPUT(_is_dof);

        TileToConAsync<float>(_a_diag, tile_dim_, *(amgpcg_.poisson_vector_[0].a_diag_), sim_stream);
        TileToConAsync<float>(_a_x, tile_dim_, *(amgpcg_.poisson_vector_[0].a_x_), sim_stream);
        TileToConAsync<float>(_a_y, tile_dim_, *(amgpcg_.poisson_vector_[0].a_y_), sim_stream);
        TileToConAsync<float>(_a_z, tile_dim_, *(amgpcg_.poisson_vector_[0].a_z_), sim_stream);
        TileToConAsync<uint8_t>(_is_dof, tile_dim_, *(amgpcg_.poisson_vector_[0].is_dof_), sim_stream);
        cudaStreamSynchronize(sim_stream);
    }
    void load_rhs(torch::Tensor _b)
    {
        CHECK_INPUT(_b);

        ConToTileAsync<float>(*(amgpcg_.b_), tile_dim_, _b, sim_stream);
        cudaStreamSynchronize(sim_stream);
    }
    void extract_rhs(torch::Tensor _b)
    {
        CHECK_INPUT(_b);

        TileToConAsync<float>(_b, tile_dim_, *(amgpcg_.b_), sim_stream);
        cudaStreamSynchronize(sim_stream);
    }
    void build(float _default_a_diag, float _default_a_off_diag)
    {
        amgpcg_.BuildAsync(_default_a_diag, _default_a_off_diag, sim_stream);
        cudaStreamSynchronize(sim_stream);
    }
    void solve(torch::Tensor _x, bool _pure_neumann = false, float _rel_tol = 1e-12, float _abs_tol = 1e-12, int _max_iter = 400)
    {
        CHECK_INPUT(_x);

        amgpcg_.pure_neumann_ = _pure_neumann;

        amgpcg_.rel_tol_ = _rel_tol;
        amgpcg_.abs_tol_ = _abs_tol;
        amgpcg_.max_iter_ = _max_iter;

        amgpcg_.SolveAsync(sim_stream);
        cudaStreamSynchronize(sim_stream);
        TileToConAsync<float>(_x, tile_dim_, *(amgpcg_.x_), sim_stream);
        cudaStreamSynchronize(sim_stream);
    }
    int3 tile_dim_;
    bool verbose_ = false;
    AMGPCG amgpcg_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<AMGPCGTorch>(m, "AMGPCGTorch")
        .def(py::init<std::vector<int>, int, bool, bool, bool>(), py::arg("tile_dim_vec"), py::arg("bottom_smoothing") = 10,
            py::arg("verbose") = false, py::arg("iter_info") = false, py::arg("trim_info") = false)
        .def("load_coeff", &AMGPCGTorch::load_coeff, py::arg("a_diag"), py::arg("a_x"),
            py::arg("a_y"), py::arg("a_z"), py::arg("is_dof"))
        .def("load_rhs", &AMGPCGTorch::load_rhs, py::arg("b"))
        .def("extract_coeff", &AMGPCGTorch::extract_coeff, py::arg("a_diag"), py::arg("a_x"),
            py::arg("a_y"), py::arg("a_z"), py::arg("is_dof"))
        .def("extract_rhs", &AMGPCGTorch::extract_rhs, py::arg("b"))
        .def("build", &AMGPCGTorch::build, py::arg("default_a_diag"),
            py::arg("default_a_off_diag"))
        .def("solve", &AMGPCGTorch::solve, py::arg("x"), py::arg("pure_neumann") = false,
            py::arg("rel_tol") = 1e-12, py::arg("abs_tol") = 1e-12, py::arg("max_iter") = 400);
}