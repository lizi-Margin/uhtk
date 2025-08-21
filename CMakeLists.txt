cmake_minimum_required (VERSION 4.0.0)
project ("uhtkcpp")

# System information checks
message(STATUS "System checks:")
message(STATUS "  System name: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  System version: ${CMAKE_SYSTEM_VERSION}")

# Compiler checks
message(STATUS "Compiler checks:")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.0")
        message(WARNING "GCC version < 7.0 may have incomplete C++17 support")
    endif()
endif()

# CPU core count
include(ProcessorCount)
ProcessorCount(NUM_CORES)
message(STATUS "  CPU cores: ${NUM_CORES}")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++17 HAS_CXX17)
if(NOT HAS_CXX17)
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support")
endif()

# Static Link:
if(MSVC)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEGUG} /MTd")
else()
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
	set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
	set(BUILD_SHARED_LIBS OFF)
endif()

message("Enter subdirectory uhtk")
add_subdirectory ("c3utils")
add_subdirectory ("csv_drawer")
