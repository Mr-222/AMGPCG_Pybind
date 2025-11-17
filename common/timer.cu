#include "timer.h"
#include "core/tool/logger.h"

#include "iostream"


GPUTimerScope::GPUTimerScope(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop)
    : m_stream(stream), m_startEvent(start), m_stopEvent(stop), m_valid(true) {
    // Record the "start" event immediately upon creation
    cudaEventRecord(m_startEvent, m_stream);
}

GPUTimerScope::GPUTimerScope()
    : m_stream(nullptr), m_startEvent(nullptr), m_stopEvent(nullptr), m_valid(false) {
}

GPUTimerScope::~GPUTimerScope() {
    if (m_valid) {
        cudaEventRecord(m_stopEvent, m_stream);
    }
}

GPUTimer::GPUTimer(uint32_t maxScopes)
    : m_maxScopes(maxScopes), m_currentScope(0) {
    m_events.resize(maxScopes * 2);
    m_scopeNames.reserve(maxScopes);

    for (size_t i = 0; i < m_events.size(); ++i) {
        cudaError_t err = cudaEventCreate(&m_events[i]);
        if (err != cudaSuccess) {
            ERROR_ALL("Failed to create CUDA event index " + std::to_string(i));
            // Cleanup previously created events
            for (size_t j = 0; j < i; ++j) {
                // Fixed typo: was i < i
                cudaEventDestroy(m_events[j]);
            }
            m_events.clear();
            break;
        }
    }
}

GPUTimer::~GPUTimer() {
    for (cudaEvent_t event: m_events) {
        cudaEventDestroy(event);
    }
}

void GPUTimer::beginFrame() {
    m_currentScope = 0;
    m_scopeNames.clear();
}

GPUTimerScope GPUTimer::profileScope(cudaStream_t stream, const std::string &name) {
    if (m_currentScope >= m_maxScopes) {
        // Return dummy scope if we exceeded max scopes
        return GPUTimerScope();
    }

    uint32_t startIndex = m_currentScope * 2;
    uint32_t stopIndex = startIndex + 1;

    m_scopeNames.push_back(name);
    m_currentScope++;

    return GPUTimerScope(stream, m_events[startIndex], m_events[stopIndex]);
}

void GPUTimer::printResults(bool synchronize) const {
    if (m_currentScope == 0) {
        INFO_ALL("--- CUDA Timer (No data) ---");
        return;
    }

    if (synchronize) {
        // Synchronize on the *last* event recorded.
        // This ensures all preceding events in the stream are also complete.
        cudaEvent_t lastEvent = m_events[m_currentScope * 2 - 1];
        cudaEventSynchronize(lastEvent);
    }

    INFO_ALL("\n--- CUDA Profiler results ---");
    for (uint32_t i = 0; i < m_currentScope; ++i) {
        cudaEvent_t startEvent = m_events[i * 2];
        cudaEvent_t stopEvent = m_events[i * 2 + 1];

        float milliseconds = 0.0f;
        cudaError_t err = cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

        if (err != cudaSuccess) {
            ERROR_ALL("Failed to get elapsed time for CUDA Timer scope: " + m_scopeNames[i]);
            continue;
        }

        INFO_ALL(m_scopeNames[i] + ": " + std::to_string(milliseconds) + " ms");
    }
    INFO_ALL("------------------------------");
}