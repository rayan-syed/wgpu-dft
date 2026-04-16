# WebGPU FFT Implementation

## Overview

`wgpu-fft` is a high-performance implementation of the Fourier Transform utilizing WebGPU. The project leverages GPU acceleration to perform efficient transform computations, making it suitable for applications in signal processing, image analysis, and other computational tasks requiring Fourier analysis.

The intention of this project was to integrate it into our [WebGPU IDT Model](https://github.com/bu-cisl/wgpu-ssnp) project. However, given the significance and broader applicability of this module, we decided to make it a standalone repository so that others can also benefit from it.

## Features

- **GPU Acceleration**: Efficient Fourier transform computation using WebGPU for parallel processing
- **Automatic Routing**: Uses the FFT for power-of-2 dimensions and falls back to the DFT otherwise
- **Inverse Transform**: Supports computation of the inverse transform via an input flag
- **Device-Agnostic**: Compatible with various GPU and compute backends, not tied to a specific platform or vendor
- **Web Integration**: Can be integrated with web-based applications using WebGPU support

## Implementation Details

This implementation utilizes a **row-wise followed by column-wise traversal** approach to compute the Fourier transform. The top-level API is exposed through `fft(...)`, which automatically dispatches to the Cooley-Tukey FFT for power-of-2 input dimensions and otherwise falls back to the direct DFT implementation. This keeps the public interface minimal while still supporting arbitrary matrix sizes.

Both implementations use a two-pass strategy. The transform is first computed along each row of the input matrix, enabling parallel processing across rows, and is then computed along each column. For power-of-2 inputs, the FFT path provides the expected performance advantage, while the DFT path remains available for non-power-of-2 dimensions or for cases where the direct method is preferred.

It is well known that GPU-based computations can be prone to inaccuracies. To mitigate this, we incorporated several optimizations within the shader files to improve numerical precision. 


