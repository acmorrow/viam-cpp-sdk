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

# TODO: [required] Buf (re)-generation
# TODO: [required] Fixup viam_rust_utils in install dir
# TODO: [urgent] Header installation
# TODO: [urgent] libviam_rust_utils handling
# TODO: [urgent] Finding external projects vs building in (e.g grpc)
# TODO: Runtime paths and make BUILD_SHARED_LIBS work
# TODO: CMake version minimum (would like to do latest, but can we)
# TODO: Check required versions of third party libraries
# TODO: Compiler minima
# TODO: Better clang-format integration (versions, dependencies, only changed, etc.)
# TODO: Version from git describe?


# Require CMake 3.18, which is the system version in Debian Bullseye.
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)


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


# Produce GNU-style variables for installation, and default
# the installation directory to be local to the build. If you
# intend to install elsewhere, pass an explicit argument
# to CMAKE_INSTALL_PREFIX on the command line.
include(GNUInstallDirs)
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()


# Manage the rust utils library. For now, we require a
# `viam_rust_utils` in the root of the tree. The user is responsible
# for providing it. We then stage it into the installation.
#
# TODO: Error out if this glob doesn't match anything
# TODO: Consider downloading this library for the user if not present
# TODO: Make it `IMPORT`ed
# TODO: Fixup runtime paths
file(GLOB viam_rust_utils_lib ${PROJECT_SOURCE_DIR}/libviam_rust_utils*.*)
install(
  FILES ${viam_rust_utils_lib}
  TYPE LIB
  COMPONENT runtime
)


# Install the license file
install(FILES
  LICENSE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/${CMAKE_PROJECT_NAME}
  COMPONENT runtime
)


# Add a `format` target. This is not a great way to go about it, but
# it is enough to get started and match what the existing Makefile
# based setup did. A better way (perhaps) would be to add
# `clang-format` as an `IMPORT`ed executable and make this target
# dependent on it (and, probably, any .clang-format files). Then use
# CMake to find the sources rather than a shell script, and add them
# as dependencies so that the `format` target is a no-op if nothing is
# out of date.
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
