#define WEBGPU_CPP_IMPLEMENTATION
#include "fft/fft.h"
#include "webgpu_utils.h"
#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

namespace {

bool parseForceDft(int argc, char* argv[]) {
    for (int index = 1; index < argc; ++index) {
        const string arg(argv[index]);
        if (arg == "--force-dft" || arg == "dft") {
            return true;
        }
    }
    return false;
}

vector<complex<float>> flattenMatrix(const vector<vector<complex<float>>>& input) {
    vector<complex<float>> flatInput;
    flatInput.reserve(input.size() * input.front().size());
    for (const auto& row : input) {
        flatInput.insert(flatInput.end(), row.begin(), row.end());
    }
    return flatInput;
}

void printMatrix(const vector<float>& buffer, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const int index = (row * cols + col) * 2;
            cout << buffer[index] << " " << buffer[index + 1];
            if (col < cols - 1) {
                cout << " ";
            }
        }
        cout << "\n";
    }
}

} // namespace

int main(int argc, char* argv[]) {
    const bool forceDft = parseForceDft(argc, argv);

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

    vector<complex<float>> flatInput = flattenMatrix(input);

    WebGPUContext context;
    initWebGPU(context);

    const int total = rows * cols;
    wgpu::Buffer forwardBuffer = createBuffer(context.device, nullptr, sizeof(float) * 2 * total,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    wgpu::Buffer inverseBuffer = createBuffer(context.device, nullptr, sizeof(float) * 2 * total,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)); 

    wgpu::Buffer inputBuffer = createBuffer(context.device, flatInput.data(), sizeof(float) * 2 * flatInput.size(), 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));

    fft(context, forwardBuffer, inputBuffer, flatInput.size(), rows, cols, 0, forceDft);
    vector<float> forwardOutput = readBack(context.device, context.queue, 2 * total, forwardBuffer);

    fft(context, inverseBuffer, inputBuffer, flatInput.size(), rows, cols, 1, forceDft);
    vector<float> inverseOutput = readBack(context.device, context.queue, 2 * total, inverseBuffer);

    cout << rows << " " << cols << "\n";
    printMatrix(forwardOutput, rows, cols);
    printMatrix(inverseOutput, rows, cols);

    wgpuQueueRelease(context.queue);
    wgpuDeviceRelease(context.device);
    wgpuAdapterRelease(context.adapter);
    wgpuInstanceRelease(context.instance);
    forwardBuffer.release();
    inverseBuffer.release();
    inputBuffer.release();

    return 0;
}
