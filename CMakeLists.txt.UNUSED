cmake_minimum_required(VERSION 3.14)

################################################################################
# Import additional CMake modules.
################################################################################
include(CheckIPOSupported)

################################################################################
# Define the project.
################################################################################
project(RiptideCPP C CXX)

################################################################################
# Determine the filename suffix and extension to use; the compiled extension (library)
# will have this appended to the name so it's recognized as a CPython extension.
################################################################################
execute_process(
  COMMAND "python" -c "import importlib.machinery; print(importlib.machinery.EXTENSION_SUFFIXES[0], end='')"
    RESULT_VARIABLE _PYTHON_EXTENSION_SUFFIX_RESULT
    OUTPUT_VARIABLE _PYTHON_EXTENSION_SUFFIX
    ERROR_QUIET)
if(NOT _PYTHON_EXTENSION_SUFFIX_RESULT EQUAL "0")
  message(WARNING "Cannot deduce the python C extension file suffix, fall back to default, set LIB_SUFFIX to override")
  if(WIN32)
    set(_PYTHON_EXTENSION_SUFFIX ".lib")
  else(WIN32)
    set(_PYTHON_EXTENSION_SUFFIX ".so")
  endif(WIN32)
endif(NOT _PYTHON_EXTENSION_SUFFIX_RESULT EQUAL "0")

set(LIB_SUFFIX ${_PYTHON_EXTENSION_SUFFIX} CACHE STRING "The suffix of the C extension library")

################################################################################
# Find Python and NumPy libraries and headers needed to build
# the riptide_cpp extension.
################################################################################
find_package(Threads REQUIRED)
find_package(Python REQUIRED COMPONENTS Development NumPy)

################################################################################
# Pull in third-party / external source libraries.
# Override build options for some libraries e.g. so we target the correct
# architecture or avoid building parts we don't need/want.
################################################################################
## crc32c
# TODO: Build this with ExternalProject_Add instead? Even if we keep our vendored copy instead of having the CMake script
#       download the source, this may provide an easier way of specifying the necessary property overrides without affecting
#       the rest of the riptide build (e.g. the requirement to set BUILD_SHARED_LIBS).
#       Related: https://github.com/google/crc32c/pull/34
set(CRC32C_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CRC32C_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(CRC32C_USE_GLOG OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # Hack: this is a global setting
if(UNIX AND NOT APPLE)
  set(BUILD_SHARED_LIBS OFF)     # Hack to fix compilation failing due to missing -fPIC when crc32c is a static library
endif()
add_subdirectory(CPP/third_party/crc32c)
if(UNIX AND NOT APPLE)
  set(BUILD_SHARED_LIBS ON)
endif()

## zstd
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
#set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
# We only use the static library for zstd, so don't bother building the shared lib.
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
# Turn off multithreading support -- we have our own multithreading in riptide.
set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE BOOL "" FORCE)
add_subdirectory(CPP/third_party/zstd/build/cmake)

################################################################################
# Define the riptide_cpp library; add source files, set properties, then add
# the library to the project.
################################################################################
file(GLOB CPP_SRC_FILES CONFIGURE_DEPENDS CPP/riptide_cpp/*.cpp)
  
# Add the riptide_cpp library target to the project.
add_library(riptide_cpp SHARED ${CPP_SRC_FILES} ${C_SRC_FILES})

# Add additional #include directories to make necessary headers available.
target_include_directories(riptide_cpp PRIVATE
  ./CPP/riptide_cpp
  ./CPP/third_party
  ./CPP/third_party/zstd/lib
)
  
# Set libraries to link with.
target_link_libraries(riptide_cpp
  Threads::Threads
  Python::Python
  Python::NumPy
  crc32c
  libzstd_static
)

# Set C++ version requirements to make sure the compiler supports it.
set_target_properties(riptide_cpp PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)

# Set the library suffix for the project to the CPython suffix we determined.
set_target_properties(riptide_cpp PROPERTIES PREFIX "" SUFFIX ${LIB_SUFFIX})

# Setting the target properties doesn't cause the corresponding .pdb to be renamed to
# match, which then causes the .pdb not to be considered part of the output by the 'install' target.
# Fix this so the .pdb is available in the installed Python egg.
if (WIN32)
  get_filename_component(_PYTHON_EXTENSION_SUFFIX_NOEXT ${_PYTHON_EXTENSION_SUFFIX} NAME_WLE)
  
  # Create the PDB filename by taking the "normal" output name of the library and appending
  # the CPython-extension suffix (without filename extension).
  #get_target_property(RIPTIDE_OUTPUT_NAME riptide_cpp LIBRARY_OUTPUT_NAME) # TODO: Use this instead to ensure consistency; this needs to be fixed though, it gives us RIPTIDE_OUTPUT_NAME-NOTFOUND
  set(RIPTIDE_OUTPUT_NAME "riptide_cpp")
  string(CONCAT _PYTHON_EXTENSION_PDB_NAME ${RIPTIDE_OUTPUT_NAME} ${_PYTHON_EXTENSION_SUFFIX_NOEXT})
  
  set_target_properties(riptide_cpp PROPERTIES PDB_NAME ${_PYTHON_EXTENSION_PDB_NAME})
endif()

################################################################################
# Set compiler-specific options/flags.
################################################################################
if (MSVC)
  # Enable some diagnostic warnings (/W3), but exclude C4100 -- those should be looked when we have time
  # but for now it generates so much noise (due to templating) it's hard to see the other warnings.
  set_source_files_properties(${CPP_SRC_FILES} PROPERTIES COMPILE_FLAGS
    PROPERTIES COMPILE_FLAGS "/arch:AVX2 /W3 /wd4100")
  set_source_files_properties(${C_SRC_FILES} PROPERTIES COMPILE_FLAGS
    PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    
  # When building with VS2017 or newer, add the /permissive- flag when compiling C++ source files
  # to enforce stronger checks for conformance to the C++ language standard.
  if(MSVC_VERSION GREATER_EQUAL 1910)
    set_property(SOURCE ${CPP_SRC_FILES} APPEND_STRING PROPERTY COMPILE_FLAGS " /permissive-")
  endif()
else () # MSVC
  # TODO: Remove the -fpermissive here -- it would be better to enforce standard-conformance rather than just warning about issues.
  set_source_files_properties(${CPP_SRC_FILES} PROPERTIES COMPILE_FLAGS
      "-march=core-avx2 -mavx2 -mbmi2 -mlzcnt -Wno-unused-variable -falign-functions=32 -falign-loops=32 -fpermissive")
  set_source_files_properties(${C_SRC_FILES} PROPERTIES COMPILE_FLAGS
      "-march=core-avx2 -mavx2 -mbmi2 -mlzcnt -Wno-unused-variable -falign-functions=32 -falign-loops=32")
endif()

################################################################################
# Set some additional optimization options (compiler-specific, but that's handled by CMake).
################################################################################
# Optional IPO. Do not use IPO if it's not supported by compiler.
# If compiling with MSVC, this is meant to enable the use of Link-Time Code Generation (LTCG).
check_ipo_supported(RESULT ipo_is_supported OUTPUT error_text_if_any)
if(ipo_is_supported)
  set_property(TARGET riptide_cpp PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
  message(WARNING "IPO is not supported: ${error_text_if_any}")
endif()

################################################################################
# Preserve debugging information.
################################################################################
# The RelWithDebInfo configuration passes compiler flags to preserve/emit debugging info
# but also uses /O2 (MSVC) or -O2 (gcc) rather than /Ox (MSVC) or -O3 (gcc).
# We want to keep the max. optimization flags but turn on debugging info so we get better
# symbols / stacktraces in case there's a crash. Do this by adding the compiler-specific flag
# for this regardless of build configuration.
if (MSVC)
  target_compile_options(riptide_cpp PUBLIC /Zi)
  target_link_options(riptide_cpp PUBLIC /DEBUG)
else() # MSVC
  # Assuming gcc/clang here
  target_compile_options(riptide_cpp PUBLIC -g)
endif()

################################################################################
# Copy the compiled targets (binaries) to the destination folder.
################################################################################
if(WIN32)
  install(TARGETS riptide_cpp RUNTIME DESTINATION ./)
  
  # Install the corresponding .pdb file as well so we have debug information available
  # in the installed package.
  install(FILES $<TARGET_PDB_FILE:riptide_cpp> DESTINATION ./)
else(WIN32)
  install(TARGETS riptide_cpp LIBRARY DESTINATION ./)
endif(WIN32)
