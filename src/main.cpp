#define WEBGPU_CPP_IMPLEMENTATION
#include "dft/dft.h"
#include "webgpu_utils.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <complex>
#include <string>

using namespace std;

int main() {

    // read input matrix
    ifstream infile("input.txt");
    if (!infile) {
        cerr << "Failed to open input.txt" << endl;
        return -1;
    }

    int rows, cols;
    infile >> rows >> cols;
    vector<vector<complex<float>>> input(rows, vector<complex<float>>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float re, im;
            infile >> re >> im;
            input[i][j] = {re,im};
        }
    }
    infile.close();

    // Flatten matrix row-major order for dft input
    vector<complex<float>> flatInput;
    flatInput.reserve(rows * cols);
    for (const auto &row : input) {
        flatInput.insert(flatInput.end(), row.begin(), row.end());
    }

    // init wgpu
    WebGPUContext context;
    initWebGPU(context);

    // init output buffer
    int total = rows * cols;
    wgpu::Buffer dftBuffer = createBuffer(context.device, nullptr, sizeof(float) * 2 * total,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    wgpu::Buffer idftBuffer = createBuffer(context.device, nullptr, sizeof(float) * 2 * total,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)); 
    
    // compute 2D DFT
    dft(context, dftBuffer, flatInput, rows, cols, 0);
    vector<float> dft_out = readBack(context.device, context.queue, 2 * total, dftBuffer);

    // compute 2D IDFT
    dft(context, idftBuffer, flatInput, rows, cols, 1);
    vector<float> idft_out = readBack(context.device, context.queue, 2 * total, idftBuffer);

    // reform output into a 2D matrix
    vector<vector<std::complex<float>>> result(rows, vector<std::complex<float>>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int index = (i * cols + j) * 2;
            result[i][j] = { dft_out[index], dft_out[index + 1] };
        }
    }

    vector<vector<std::complex<float>>> iresult(rows, vector<std::complex<float>>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int index = (i * cols + j) * 2;
            iresult[i][j] = { idft_out[index], idft_out[index + 1] };
        }
    }

    // print output for test script to read in
    cout << rows << " " << cols << "\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << result[i][j].real() << " " << result[i][j].imag();
            if (j < cols - 1) cout << " ";
        }
        cout << "\n";
    }

    cout << rows << " " << cols << "\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << iresult[i][j].real() << " " << iresult[i][j].imag();
            if (j < cols - 1) cout << " ";
        }
        cout << "\n";
    }



    // clear wgpu resources
    wgpuQueueRelease(context.queue);
    wgpuDeviceRelease(context.device);
    wgpuAdapterRelease(context.adapter);
    wgpuInstanceRelease(context.instance);
    dftBuffer.release();
    idftBuffer.release();

    return 0;
}