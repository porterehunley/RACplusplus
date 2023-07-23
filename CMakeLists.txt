cmake_minimum_required(VERSION 3.16)
project(racplusplus VERSION "0.0.5")
message(STATUS "CC: $ENV{CC}")
message(STATUS "CXX: $ENV{CXX}")

# Set the OpenMP flags here
if(DEFINED ENV{OpenMP_LIBRARY})
  message(STATUS "OpenMP_LIBRARY = $ENV{OpenMP_LIBRARY}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")
  include_directories("/usr/local/include")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "The C++ compiler is g++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# Set Python3.11 directories
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
list(APPEND CMAKE_PREFIX_PATH ${Python_SITEARCH})
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

message(STATUS "-- -- Found Python ${Python3_VERSION}")
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

# Set compilers
message(STATUS "\n-- Setting C and C++ Compilers to llvm...\n")
# print cc and cxx env variables

set(CMAKE_CXX_COMPILER "$ENV{CXX}")
set(ENV{CPPFLAGS} "-I${LOCAL_INCLUDE_DIRS}")

# Print out compiler information
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

#Set Opt, Warning Flags, Linker
message(STATUS "\n-- Setting C++ Warning and Optimization Flags, Setting Linker to lld...\n")

if (MSVC)
  message(STATUS "MSVC detected, setting flags...")
  add_compile_options(/openmp)

  # Get the Python version
  execute_process(COMMAND python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
  OUTPUT_VARIABLE PYTHON_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)

  message(STATUS "PYTHON_VERSION: ${PYTHON_VERSION}")

  list(APPEND CMAKE_PREFIX_PATH "C:/hostedtoolcache/windows/Python/${PYTHON_VERSION}/x64/lib/site-packages/pybind11")
  add_compile_options(/O2 /W4 /EHsc)
  add_compile_options(/std:c++latest)

else()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wsign-compare -Wunreachable-code -fwrapv -Wall")
  set(CMAKE_CXX_STANDARD 17)
endif()

#-------------------required packages-------------------------
find_package(Python3 REQUIRED Development.Module)
find_package(Python3 REQUIRED Interpreter)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
#-------------------------------------------------------------

#-------------------------------------Include Directories----------------------------
#include python directory for Python.h header file
include_directories(${PYTHON311_INCLUDE_DIRS})
# Use the pybind11_INCLUDE_DIRS variable to access the include directories
include_directories(${pybind11_INCLUDE_DIRS})
#include eigen directories
include_directories(${EIGEN3_INCLUDE_DIRS} )
#------------------------------------------------------------------------------------

#-----------------------------------Make Library---------------------------------------
set(python_module_name _racplusplus)
pybind11_add_module(${python_module_name} MODULE
     src/racplusplus/_racplusplus.cpp
    )

#-----------------------------------Link Libraries-------------------------------------
if (MSVC)
  target_link_libraries(${python_module_name} PRIVATE
      -L${PYTHON311_LIBRARIES}
      ${pybind11_LIBRARIES}
      Eigen3::Eigen)
else()
  target_link_libraries(${python_module_name} PRIVATE
      -L${PYTHON311_LIBRARIES}
      ${pybind11_LIBRARIES}
      c++
      Eigen3::Eigen)
endif()
#----------------------------------------------------------------------------------------

#Install the Cmake Target in src directory specified in setup.py
install(TARGETS ${python_module_name}
        DESTINATION .) 