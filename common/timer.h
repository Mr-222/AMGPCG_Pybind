#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>

class GPUTimer;

class GPUTimerScope {
public:
    friend class GPUTimer;

    // Valid scope constructor
    GPUTimerScope(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop);

    // Invalid ("dummy") scope constructor
    GPUTimerScope();

    ~GPUTimerScope();

    // Movable but not copyable
    GPUTimerScope(const GPUTimerScope &) = delete;

    GPUTimerScope &operator=(const GPUTimerScope &) = delete;

    GPUTimerScope(GPUTimerScope &&) = default;

    GPUTimerScope &operator=(GPUTimerScope &&) = default;

private:
    cudaStream_t m_stream;
    cudaEvent_t m_startEvent;
    cudaEvent_t m_stopEvent;
    bool m_valid;
};

/**
 * @class GPUTimer
 * @brief Manages CUDA event query collection and reporting.
 */
class GPUTimer {
public:
    explicit GPUTimer(uint32_t maxScopes = 64);

    ~GPUTimer();

    GPUTimer(const GPUTimer &) = delete;

    GPUTimer &operator=(const GPUTimer &) = delete;

    void beginFrame();

    GPUTimerScope profileScope(cudaStream_t stream, const std::string &name);

    void printResults(bool synchronize = true) const;

private:
    uint32_t m_maxScopes;
    uint32_t m_currentScope;

    std::vector<std::string> m_scopeNames;
    std::vector<cudaEvent_t> m_events;
};

#define CUDA_PROFILE_SCOPE(profiler, stream, name) auto profile_scope_##__LINE__ = (profiler).profileScope(stream, name);
