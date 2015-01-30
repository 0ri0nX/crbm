if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS -DCUDA)
    CUDA_ADD_LIBRARY(crbmComputerGpu SHARED crbmComputer.cu "${CMAKE_CURRENT_SOURCE_DIR}/../matrixCpu.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../utils.cpp" crbmComputerBinding.cpp)
    #set_target_properties(crbmComputeGpu PROPERTIES COMPILE_DEFINITIONS CUDA)
endif()

add_library(crbmComputerCpu SHARED crbmComputer.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../matrixCpu.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../utils.cpp" crbmComputerBinding.cpp)
add_library(crbmComputerCpuStatic STATIC crbmComputer.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../matrixCpu.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../utils.cpp" crbmComputerBinding.cpp)
set_target_properties(crbmComputerCpu crbmComputerCpuStatic PROPERTIES COMPILE_DEFINITIONS SQUEEZE)

target_link_libraries(crbmComputerCpu blas rt)

set(PC_NAME "")
set(PC_DESCRIPTION "")
set(PC_VERSION "")
set(PC_URL "")
set(PC_REQUIRES "")
set(PC_REQUIRES_PRIVATE "")
set(PC_CONFLICTS "")


set(PC_LIBS_FULL "-lcrbmComputerCpu")
configure_file(pkg-config.pc.in crbmComputerCpu.pc @ONLY)

set(PC_LIBS_FULL "-lcrbmComputerCpuStatic -lblas -lrt")
configure_file(pkg-config.pc.in crbmComputerCpuStatic.pc @ONLY)

#api version, SOVERSION == VERSION[0]
set_target_properties(crbmComputerCpu PROPERTIES SOVERSION 0 VERSION 0.0.0)

install(TARGETS crbmComputerCpu crbmComputerCpuStatic
    DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES crbmComputerBinding.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/crbm")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/crbmComputerCpu.pc ${CMAKE_CURRENT_BINARY_DIR}/crbmComputerCpuStatic.pc
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

configure_file(test/runtest.sh runtest.sh @ONLY)

add_test(runtest ${CMAKE_CURRENT_BINARY_DIR}/runtest.sh)

