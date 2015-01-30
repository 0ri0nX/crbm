# Minimální potřebná verze cmake
# Nejlépe nastavit na stejnou verzi cmake, ktera je nainstalována.
cmake_minimum_required(VERSION 2.8.2)

# začátek nového projektu
project(crbm)
enable_testing()

set(CMAKE_CXX_FLAGS "-Wall -std=c++0x ${CMAKE_CXX_FLAGS}")

# fulltext rozšiření z baliku szn-fulltext-cmake
#find_package(fulltext-cmake 0.1.6 REQUIRED)
find_package(CUDA)
if (CUDA_FOUND)
    message ("Cuda found ..")
else()
    message ("Cuda not found ..")
endif()

#include(GNUInstallDirs)
set(CMAKE_INSTALL_LIBDIR "lib")
set(CMAKE_INSTALL_INCLUDEDIR "include")

add_subdirectory(src)


