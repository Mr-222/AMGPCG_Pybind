#pragma once

#include "trim_poisson.h"
#include <vector>

namespace ofm {
class AMGPCG {
public:
    int3 tile_dim_;
    int level_num_;

    bool pure_neumann_    = true;
    int bottom_smoothing_ = 10;
    bool verbose_         = false;
    bool cooperative_GS_  = false;
    bool iter_info_       = false;
    bool trim_info_       = false;
    bool solve_by_tol_    = false;
    float rel_tol_        = 1e-12;
    float abs_tol_        = 1e-14;
    int max_iter_         = 400;

    std::shared_ptr<DHMemory<float>> rTr_;
    std::shared_ptr<DHMemory<float>> old_zTr_;
    std::shared_ptr<DHMemory<float>> new_zTr_;
    std::shared_ptr<DHMemory<float>> pAp_;
    std::shared_ptr<DHMemory<float>> alpha_;
    std::shared_ptr<DHMemory<float>> beta_;
    std::shared_ptr<DHMemory<float>> avg_;
    std::shared_ptr<DHMemory<int>> num_dof_;

    std::shared_ptr<DHMemory<float>> x_;
    std::shared_ptr<DHMemory<float>> p_;
    std::shared_ptr<DHMemory<float>> Ap_;
    std::shared_ptr<DHMemory<float>> b_;
    std::shared_ptr<DHMemory<float>> dot_buffer_;
    std::shared_ptr<DHMemory<uint8_t>> dot_tmp_;
    std::shared_ptr<DHMemory<int>> block_num_dof_;
    std::shared_ptr<DHMemory<uint8_t>> block_num_dof_tmp_;

    std::vector<TrimPoisson> poisson_vector_;

    std::shared_ptr<DHMemory<float>> mul_;
    std::shared_ptr<DHMemory<uint8_t>> global_dot_tmp_;

    AMGPCG() = default;
    AMGPCG(int3 _tile_dim, int _level_num);
    void Alloc(int3 _tile_dim, int _level_num);

    void BuildAsync(float _default_a_diag, float _default_a_off_diag, cudaStream_t _stream);

    void ProlongateAsync(int _coarse_level, cudaStream_t _stream);
    void VcycleDotAsync(cudaStream_t _stream);
    void AxpyAsync(std::shared_ptr<DHMemory<float>> _dst, const std::shared_ptr<DHMemory<float>> _coef,
                   std::shared_ptr<DHMemory<float>> _x, std::shared_ptr<DHMemory<float>> _y, cudaStream_t _stream);
    void YmAxAsync(std::shared_ptr<DHMemory<float>> _dst, const std::shared_ptr<DHMemory<float>> _coef,
                   std::shared_ptr<DHMemory<float>> _x, cudaStream_t _stream);
    void DotSumAsync(std::shared_ptr<DHMemory<float>> _dst, cudaStream_t _stream);
    void SolveAsync(cudaStream_t _stream);

    void RecenterAsync(std::shared_ptr<DHMemory<float>> _dst, cudaStream_t _stream);
    void CountDofAsync(cudaStream_t _stream);

    void DotAsync(std::shared_ptr<DHMemory<float>> _dst, const std::shared_ptr<DHMemory<float>> _src1,
                  const std::shared_ptr<DHMemory<float>> _src2, cudaStream_t _stream);
};
};