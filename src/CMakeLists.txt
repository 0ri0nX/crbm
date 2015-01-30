include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(CUDA_FOUND)
    #add_definitions(-DCUDA)
    cuda_add_executable(crbmGpu crbmGpu.cu matrixCpu.cpp utils.cpp )
    cuda_add_executable(crbmComputeGpu crbmComputeGpu.cu matrixCpu.cpp utils.cpp )

    target_link_libraries(crbmGpu ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} blas rt)
    target_link_libraries(crbmComputeGpu ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} blas rt)

    #-lcuda -lcublas -lcurand -lcudart
    #remove_definitions(-DCUDA)
endif()

add_executable(crbmComputeCpu crbmComputeCpu.cpp matrixCpu.cpp utils.cpp)
target_link_libraries(crbmComputeCpu blas rt)
set_target_properties(crbmComputeCpu PROPERTIES COMPILE_DEFINITIONS SQUEEZE)

add_subdirectory(crbmComputer)


