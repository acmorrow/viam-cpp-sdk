# Copyright 2023 Viam Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: [required] libviam_rust_utils handling
# TODO: [required] Dockerfile
# TODO: Runtime paths and make BUILD_SHARED_LIBS work
# TODO: Check required versions of third party libraries
# TODO: Compiler minima
# TODO: Finding external projects vs building in (e.g grpc)
# TODO: Better clang-format integration (versions, dependencies, only changed, etc.)
# TODO: Version from git describe?
# TODO: Export macros and visibilty annotations


# This is a tricky decision. If we set this too old, we lose access to
# modern CMake facilities that we want to be able to use, like file
# sets. If we set it too new, then the system CMake can't be used to
# build the product. Here, we make an opinionated choice to require
# the very latest stable CMake as of this writing, arguing that for
# most uses of this product, it is expected to be cross-compiling for
# the target platform on a development system, meaning that we are not
# constrained by the version of CMake available on target systems.
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)


# Identify the project.
project(viam-cpp-sdk
  VERSION 0.0.0
  DESCRIPTION "Viam Robotics C++ SDK"
  HOMEPAGE_URL https://github.com/viamrobotics/viam-cpp-sdk
  LANGUAGES CXX
)


# Enforce the C++ standard, and disable extensions
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)


# Produce GNU-style variables for installation, and default the
# installation directory to be local to the build. If you intend to
# install elsewhere, pass an explicit argument to CMAKE_INSTALL_PREFIX
# on the command line:
#
# cmake ... -DCMAKE_INSTALL_PREFIX=$HOME/opt
#
include(GNUInstallDirs)
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()


# See if the user provided a libviam_rust_utils in the root of the
# tree. If not, try to download the latest.
#
# TODO: Fixup runtime paths at install time.
# TODO: This should be pushed down into a tool
file(GLOB lib_viam_rust_utils ${PROJECT_SOURCE_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}viam_rust_utils*${CMAKE_SHARED_LIBRARY_SUFFIX})
if (NOT lib_viam_rust_utils)
  set(lvru_system_name ${CMAKE_SYSTEM_NAME})
  if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(lvru_system_name "macosx")
  endif()
  file(
    DOWNLOAD
    https://github.com/viamrobotics/rust-utils/releases/latest/download/libviam_rust_utils-${lvru_system_name}_${CMAKE_SYSTEM_PROCESSOR}.dylib
    ${CMAKE_SHARED_LIBRARY_PREFIX}viam_rust_utils${CMAKE_SHARED_LIBRARY_SUFFIX}
    STATUS lvru_status
  )
  list(GET lvru_status 0 lvru_status_code)
  list(GET lvru_status 1 lvru_status_string)

  if(NOT lvru_status_code EQUAL 0)
    message(FATAL_ERROR "No local viam_rust_utils found and failed to download: ${lvru_status_string}")
  endif()

  set(lib_viam_rust_utils ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}viam_rust_utils${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

add_library(viam_rust_utils SHARED IMPORTED)
set_property(TARGET viam_rust_utils PROPERTY IMPORTED_LOCATION ${lib_viam_rust_utils})

install(
  FILES ${lib_viam_rust_utils}
  TYPE LIB
  COMPONENT runtime
  RENAME ${CMAKE_SHARED_LIBRARY_PREFIX}viam_rust_utils${CMAKE_SHARED_LIBRARY_SUFFIX}
)


# Install the license file
install(FILES
  LICENSE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/${CMAKE_PROJECT_NAME}
  COMPONENT runtime
)


# Add `format` and `check-format` targets to run clang-format or just
# check for errors. This is not a great way to go about it, but it is
# enough to get started and match what the existing Makefile based
# setup did. A better way (perhaps) would be to add `clang-format` as
# an `IMPORT`ed executable and make this target dependent on it (and,
# probably, any .clang-format files). Then use CMake to find the
# sources rather than a shell script, and add them as dependencies so
# that the `format` target is a no-op if nothing is out of date.
add_custom_target(
  format
  COMMAND ${PROJECT_SOURCE_DIR}/bin/run-clang-format.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_custom_target(
  check-format
  COMMAND ${PROJECT_SOURCE_DIR}/bin/run-clang-format.sh --dry-run -Werror
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  USES_TERMINAL
)


# Pull in our subdirectories
add_subdirectory(src)
add_subdirectory(examples)
