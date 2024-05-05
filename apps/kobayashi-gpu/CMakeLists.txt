cmake_minimum_required(VERSION 3.8)

project(KobayashiGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

add_executable(KobayashiGPU KobayashiGPU.cu)

set_target_properties(KobayashiGPU PROPERTIES CUDA_ARCHITECTURES "50;70;72")
