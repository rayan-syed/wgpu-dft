#define WEBGPU_CPP_IMPLEMENTATION
#include "dft/dft.h"
#include "webgpu_utils.h"
#include <vector>
#include <iostream>
#include <complex>

using namespace std;

int main() {
    // Initialize WebGPU
    WebGPUContext context;
    initWebGPU(context);

    // input matrix
    vector<vector<complex<float>>> input = {
        { {1.0f, 1.0f}, {2.0f, 2.0f} },
        { {3.0f, 3.0f}, {4.0f, 4.0f} }
    };

    // compute 2D dft
    int rows = input.size();
    int cols = rows > 0 ? input[0].size() : 0;
    wgpu::Buffer dftBuffer = createBuffer(context.device, nullptr, sizeof(float) * 2 * rows * cols, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    dft(context, dftBuffer, input);
    vector<float> dft_out = readBack(context.device, context.queue, 2 * rows * cols, dftBuffer);

    // reform matrix
    vector<vector<complex<float>>> result(rows, vector<complex<float>>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int index = (i * cols + j) * 2; // Real and imaginary parts
            result[i][j] = {dft_out[index], dft_out[index + 1]};
        }
    }

    // print matrix
    cout << "DFT Output (real,imag):" << endl;
    for (const auto& row : result) {
        for (const auto& val : row) {
            cout << "(" << val.real() << ", " << val.imag() << ") ";
        }
        cout << endl;
    }

    // Release WebGPU resources
    wgpuQueueRelease(context.queue);
    wgpuDeviceRelease(context.device);
    wgpuAdapterRelease(context.adapter);
    wgpuInstanceRelease(context.instance);
    dftBuffer.release();

    return 0;
}