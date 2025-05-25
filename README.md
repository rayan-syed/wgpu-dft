# WebGPU DFT Implementation

## Overview

`wgpu-dft` is a high-performance implementation of the Discrete Fourier Transform (DFT) utilizing WebGPU. The project leverages GPU acceleration to perform efficient DFT computations, making it suitable for applications in signal processing, image analysis, and other computational tasks requiring Fourier analysis.

The intention of this project was to integrate it into our [WebGPU SSNP-IDT Model](https://github.com/andrewx-bu/wgpu_ssnp-idt) project. However, given the significance and broader applicability of this module, we decided to make it a standalone repository so that others can also benefit from it.

## Features

- **GPU Acceleration**: Efficient DFT computation leveraging WebGPU for parallel processing
- **Inverse DFT**: Supports computation of the Inverse Discrete Fourier Transform (IDFT) via an input flag
- **Device-Agnostic**: Compatible with various GPU and compute backends, not tied to a specific platform or vendor
- **Web Integration**: Can be integrated with web-based applications using WebGPU support

## Implementation Details

This implementation utilizes a **row-wise followed by column-wise traversal** approach to compute the Discrete Fourier Transform (DFT). The transform is first computed along each row of the input matrix, reducing the problem size and enabling parallel processing across rows. Once the rows are transformed, the DFT is then computed along each column. This sequential two-pass strategy significantly reduces computation time compared to a brute-force approach.

While a Fast Fourier Transform (FFT) would offer even greater efficiency, we found that this DFT implementation was sufficient for our testing, even with very large inputs. In fact, the performance of our implementation was found to be **comparable to Python's built-in FFT functions**, demonstrating the efficiency of our approach despite using a direct DFT.

It is well known that GPU-based computations can be prone to inaccuracies. To mitigate this, we incorporated several optimizations within the shader files to improve numerical precision. Our tests demonstrated that the implementation maintains an accuracy within a tolerance of **1e-4**, even when processing very large input datasets.
