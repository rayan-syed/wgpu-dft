#ifndef DFT_H
#define DFT_H
#include <fstream>
#include <sstream>
#include <cmath>
#include <complex>
#include <vector>
#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"

// Performs 2D Discrete Fourier Transform (naive O(N^2) algorithm)
void dft(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t buffersize,
    int rows,
    int cols,
    uint32_t doInverse
);

#endif 
