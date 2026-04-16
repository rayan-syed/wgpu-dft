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

enum class TransformMode {
    Both,
    Forward,
    Backward,
};

bool parseForceDft(int argc, char* argv[]) {
    for (int index = 1; index < argc; ++index) {
        const string arg(argv[index]);
        if (arg == "--force-dft" || arg == "dft") {
            return true;
        }
    }
    return false;
}

TransformMode parseTransformMode(int argc, char* argv[]) {
    for (int index = 1; index < argc; ++index) {
        const string arg(argv[index]);
        if (arg == "--mode=forward" || arg == "forward") {
            return TransformMode::Forward;
        }
        if (arg == "--mode=backward" || arg == "--mode=inverse" || arg == "backward" || arg == "inverse") {
            return TransformMode::Backward;
        }
    }
    return TransformMode::Both;
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
    const TransformMode mode = parseTransformMode(argc, argv);

    ifstream infile("tests/artifacts/input.txt");
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
    wgpu::Buffer inputBuffer = createBuffer(context.device, flatInput.data(), sizeof(float) * 2 * flatInput.size(), 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));

    vector<float> forwardOutput;
    vector<float> inverseOutput;

    if (mode == TransformMode::Both || mode == TransformMode::Forward) {
        wgpu::Buffer forwardBuffer = createBuffer(context.device, nullptr, sizeof(float) * 2 * total,
            WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        fft(context, forwardBuffer, inputBuffer, flatInput.size(), rows, cols, 0, forceDft);
        forwardOutput = readBack(context.device, context.queue, 2 * total, forwardBuffer);
        forwardBuffer.release();
    }

    if (mode == TransformMode::Both || mode == TransformMode::Backward) {
        wgpu::Buffer inverseBuffer = createBuffer(context.device, nullptr, sizeof(float) * 2 * total,
            WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        fft(context, inverseBuffer, inputBuffer, flatInput.size(), rows, cols, 1, forceDft);
        inverseOutput = readBack(context.device, context.queue, 2 * total, inverseBuffer);
        inverseBuffer.release();
    }

    cout << rows << " " << cols << "\n";
    if (mode == TransformMode::Both || mode == TransformMode::Forward) {
        printMatrix(forwardOutput, rows, cols);
    }
    if (mode == TransformMode::Both || mode == TransformMode::Backward) {
        printMatrix(inverseOutput, rows, cols);
    }

    wgpuQueueRelease(context.queue);
    wgpuDeviceRelease(context.device);
    wgpuAdapterRelease(context.adapter);
    wgpuInstanceRelease(context.instance);
    inputBuffer.release();

    return 0;
}
