#define WEBGPU_CPP_IMPLEMENTATION
#include "fft/fft.h"
#include "webgpu_utils.h"
#include <algorithm>
#include <chrono>
#include <complex>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace std;

namespace {

enum class TransformMode {
    Both,
    Forward,
    Backward,
};

struct ParsedArgs {
    bool forceDft = false;
    TransformMode mode = TransformMode::Both;
    int benchmarkRepeats = 0;
};

ParsedArgs parseArgs(int argc, char* argv[]) {
    ParsedArgs args;
    for (int index = 1; index < argc; ++index) {
        const string arg(argv[index]);
        if (arg == "--force-dft" || arg == "dft") {
            args.forceDft = true;
            continue;
        }
        if (arg == "--mode=forward" || arg == "forward") {
            args.mode = TransformMode::Forward;
            continue;
        }
        if (arg == "--mode=backward" || arg == "--mode=inverse" || arg == "backward" || arg == "inverse") {
            args.mode = TransformMode::Backward;
            continue;
        }
        const string benchmarkPrefix = "--benchmark=";
        if (arg.rfind(benchmarkPrefix, 0) == 0) {
            args.benchmarkRepeats = stoi(arg.substr(benchmarkPrefix.size()));
        }
    }
    return args;
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

void runBenchmark(
    WebGPUContext& context,
    wgpu::Buffer& inputBuffer,
    size_t buffersize,
    int rows,
    int cols,
    uint32_t doInverse,
    bool forceDft,
    int repeats
) {
    vector<double> durationsMs;
    durationsMs.reserve(repeats);

    wgpu::Buffer outputBuffer = createBuffer(
        context.device,
        nullptr,
        sizeof(float) * 2 * rows * cols,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );

    for (int iteration = 0; iteration < repeats; ++iteration) {
        const auto start = chrono::steady_clock::now();
        fft(context, outputBuffer, inputBuffer, buffersize, rows, cols, doInverse, forceDft);
        waitForQueueIdle(context.device, context.queue);
        const auto end = chrono::steady_clock::now();
        durationsMs.push_back(chrono::duration<double, std::milli>(end - start).count());
    }

    outputBuffer.release();

    double sumMs = 0.0;
    double minMs = numeric_limits<double>::max();
    double maxMs = 0.0;
    for (double durationMs : durationsMs) {
        sumMs += durationMs;
        minMs = min(minMs, durationMs);
        maxMs = max(maxMs, durationMs);
    }

    cout << "benchmark\n";
    cout << "mean_ms " << (sumMs / durationsMs.size()) << "\n";
    cout << "min_ms " << minMs << "\n";
    cout << "max_ms " << maxMs << "\n";
    cout << "raw_ms";
    for (double durationMs : durationsMs) {
        cout << " " << durationMs;
    }
    cout << "\n";
}

} // namespace

int main(int argc, char* argv[]) {
    const ParsedArgs args = parseArgs(argc, argv);

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

    if (args.benchmarkRepeats > 0) {
        const uint32_t doInverse = args.mode == TransformMode::Backward ? 1 : 0;
        runBenchmark(
            context,
            inputBuffer,
            flatInput.size(),
            rows,
            cols,
            doInverse,
            args.forceDft,
            args.benchmarkRepeats
        );

        wgpuQueueRelease(context.queue);
        wgpuDeviceRelease(context.device);
        wgpuAdapterRelease(context.adapter);
        wgpuInstanceRelease(context.instance);
        inputBuffer.release();
        return 0;
    }

    vector<float> forwardOutput;
    vector<float> inverseOutput;

    if (args.mode == TransformMode::Both || args.mode == TransformMode::Forward) {
        wgpu::Buffer forwardBuffer = createBuffer(context.device, nullptr, sizeof(float) * 2 * total,
            WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        fft(context, forwardBuffer, inputBuffer, flatInput.size(), rows, cols, 0, args.forceDft);
        forwardOutput = readBack(context.device, context.queue, 2 * total, forwardBuffer);
        forwardBuffer.release();
    }

    if (args.mode == TransformMode::Both || args.mode == TransformMode::Backward) {
        wgpu::Buffer inverseBuffer = createBuffer(context.device, nullptr, sizeof(float) * 2 * total,
            WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
        fft(context, inverseBuffer, inputBuffer, flatInput.size(), rows, cols, 1, args.forceDft);
        inverseOutput = readBack(context.device, context.queue, 2 * total, inverseBuffer);
        inverseBuffer.release();
    }

    cout << rows << " " << cols << "\n";
    if (args.mode == TransformMode::Both || args.mode == TransformMode::Forward) {
        printMatrix(forwardOutput, rows, cols);
    }
    if (args.mode == TransformMode::Both || args.mode == TransformMode::Backward) {
        printMatrix(inverseOutput, rows, cols);
    }

    wgpuQueueRelease(context.queue);
    wgpuDeviceRelease(context.device);
    wgpuAdapterRelease(context.adapter);
    wgpuInstanceRelease(context.instance);
    inputBuffer.release();

    return 0;
}
