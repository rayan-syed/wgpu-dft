#ifndef DFT_H
#define DFT_H
#include <fstream>
#include <sstream>
#include <cmath>
#include <complex>
#include <vector>
#include <webgpu/webgpu.hpp>
#include "../webgpu_utils.h"

void dft(WebGPUContext& context, wgpu::Buffer& outputBuffer, std::vector<std::vector<std::complex<float>>>& input);

#endif 
