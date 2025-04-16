cmake_minimum_required(VERSION 3.0...3.25)
project(
    WEBGPU_DFT
    VERSION 0.1.0
    LANGUAGES CXX C
)

add_executable( 
    wgpu_dft
    src/main.cpp 
    src/webgpu_utils.cpp
    src/dft/dft.cpp
)

set_target_properties(wgpu_dft PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    COMPILE_WARNING_AS_ERROR ON
)

if (MSVC)
    target_compile_options(wgpu_dft PRIVATE /W4)
else()
    target_compile_options(wgpu_dft PRIVATE -Wall -Wextra -pedantic)
endif()

# Include WebGPU subdirectory
add_subdirectory(webgpu)

# Link WebGPU library to wgpu_dft
target_link_libraries(wgpu_dft PRIVATE webgpu)

# Copy necessary runtime binaries
target_copy_webgpu_binaries(wgpu_dft)