#include "dft.h"

static size_t buffer_size;

struct Params {
    int rows;
    int cols;
};

// CREATING BIND GROUP LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry inputBufferLayout = {};
    inputBufferLayout.binding = 0;
    inputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    inputBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry outputBufferLayout = {};
    outputBufferLayout.binding = 1;
    outputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    outputBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;
    
    wgpu::BindGroupLayoutEntry uniformBufferLayout = {};
    uniformBufferLayout.binding = 2;
    uniformBufferLayout.visibility = wgpu::ShaderStage::Compute;
    uniformBufferLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry entries[] = {inputBufferLayout, outputBufferLayout, uniformBufferLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 3;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

// CREATING BIND GROUP
static wgpu::BindGroup createBindGroup(wgpu::Device& device, wgpu::BindGroupLayout bindGroupLayout, wgpu::Buffer inputBuffer, wgpu::Buffer outputBuffer, wgpu::Buffer uniformBuffer) {
    wgpu::BindGroupEntry inputEntry = {};
    inputEntry.binding = 0;
    inputEntry.buffer = inputBuffer;
    inputEntry.offset = 0;
    inputEntry.size = sizeof(float) * 2 * buffer_size;

    wgpu::BindGroupEntry outputEntry = {};
    outputEntry.binding = 1;
    outputEntry.buffer = outputBuffer;
    outputEntry.offset = 0;
    outputEntry.size = sizeof(float) * 2 * buffer_size;

    wgpu::BindGroupEntry uniformEntry = {};
    uniformEntry.binding = 2;
    uniformEntry.buffer = uniformBuffer;
    uniformEntry.offset = 0;
    uniformEntry.size = sizeof(Params);
    
    wgpu::BindGroupEntry entries[] = {inputEntry, outputEntry, uniformEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void dft(WebGPUContext& context, wgpu::Buffer& finalOutputBuffer, std::vector<std::vector<std::complex<float>>>& input) {
    // Get matrix dims
    int rows = input.size();
    int cols = (rows > 0) ? input[0].size() : 0;

    // Flatten matrix row-major order - prep for row dft pass
    std::vector<std::complex<float>> flatInput;
    flatInput.reserve(rows * cols);
    for (const auto &row : input) {
        flatInput.insert(flatInput.end(), row.begin(), row.end());
    }

    buffer_size = flatInput.size();
    Params params = {rows, cols};

    // Retrieve device and queue.
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;
    WorkgroupLimits limits = getWorkgroupLimits(device);
    limits.maxWorkgroupSizeX = std::min(limits.maxWorkgroupSizeX, sqrt(limits.maxInvocationsPerWorkgroup));
    limits.maxWorkgroupSizeY = std::min(limits.maxWorkgroupSizeY, sqrt(limits.maxInvocationsPerWorkgroup));

    // Create the uniform buffer for dimensions.
    wgpu::Buffer uniformBuffer = createBuffer(device, &params, sizeof(Params), wgpu::BufferUsage::Uniform);

    // ROW DFT PASS -> save output in intermediate buffer before column pass
    wgpu::Buffer inputBuffer = createBuffer(device, flatInput.data(), sizeof(float) * 2 * buffer_size, wgpu::BufferUsage::Storage);
    wgpu::Buffer intermediateBuffer = createBuffer(device, nullptr, sizeof(float) * 2 * buffer_size, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));

    std::string shaderCodeRow = readShaderFile("src/dft/dft_row.wgsl", limits.maxWorkgroupSizeX, limits.maxWorkgroupSizeY);
    wgpu::ShaderModule shaderModuleRow = createShaderModule(device, shaderCodeRow);

    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroupRow = createBindGroup(device, bindGroupLayout, inputBuffer, intermediateBuffer, uniformBuffer);
    wgpu::ComputePipeline computePipelineRow = createComputePipeline(device, shaderModuleRow, bindGroupLayout);

    // Note: same workgroups for row pass & col pass
    uint32_t workgroupsX = std::ceil(double(cols)/16.0);
    uint32_t workgroupsY = std::ceil(double(rows)/16.0);

    wgpu::CommandBuffer commandBufferRow = createComputeCommandBuffer(device, computePipelineRow, bindGroupRow, workgroupsX, workgroupsY);
    queue.submit(1, &commandBufferRow);

    // Clean row pass resources before doing column pass
    commandBufferRow.release();
    computePipelineRow.release();
    bindGroupRow.release();
    shaderModuleRow.release();
    inputBuffer.release();

    // COLUMN DFT PASS
    std::string shaderCodeCol = readShaderFile("src/dft/dft_col.wgsl", limits.maxWorkgroupSizeX, limits.maxWorkgroupSizeY);
    wgpu::ShaderModule shaderModuleCol = createShaderModule(device, shaderCodeCol);

    wgpu::BindGroup bindGroupCol = createBindGroup(device, bindGroupLayout, intermediateBuffer, finalOutputBuffer, uniformBuffer);
    wgpu::ComputePipeline computePipelineCol = createComputePipeline(device, shaderModuleCol, bindGroupLayout);

    wgpu::CommandBuffer commandBufferCol = createComputeCommandBuffer(device, computePipelineCol, bindGroupCol, workgroupsX, workgroupsY);
    queue.submit(1, &commandBufferCol);

    // Clean all resources
    commandBufferCol.release();
    computePipelineCol.release();
    bindGroupCol.release();
    bindGroupLayout.release();
    shaderModuleCol.release();
    uniformBuffer.release();
    intermediateBuffer.release();
}
