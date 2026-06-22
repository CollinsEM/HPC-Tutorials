cmake_minimum_required(VERSION 3.10)
project(stream_triad)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add executable
add_executable(stream_triad stream_triad.cpp timer.cpp)

# Set OpenMP flags for GPU offload
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # Requires the GCC NVPTX offload plugin (e.g. the gcc-offload-nvptx package).
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -foffload=nvptx-none")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC")
    # NVIDIA HPC SDK (nvc++): -mp=gpu offloads target regions to the GPU.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mp=gpu")
endif()

# Link against OpenMP
target_link_libraries(stream_triad PRIVATE OpenMP::OpenMP_CXX)

# Add include directories
target_include_directories(stream_triad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}) 