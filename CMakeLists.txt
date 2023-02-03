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

# TODO: Header / License / viam_rust_utils intsallation
# TODO: Buf (re)-generation
# TODO: Clang-format target?
# TODO: Finding external projects
# TODO: Examples use the library
# TODO: Runtime paths
# TODO: Check required versions of third party libraries
# TODO: Compiler minima
# TODO: Version from git describe?
# TODO: ld: warning: directory not found for option '-F/Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk/System/Library/Frameworks'
# TODO: static/dynamic/both?


# Require CMake 3.18, which is the system version in Debian Bullseye.
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
endif()

project(VIAM_CPP_SDK
  VERSION 0.0.0
  DESCRIPTION "Viam Robotics C++ SDK"
  HOMEPAGE_URL https://github.com/viamrobotics/viam-cpp-sdk
  LANGUAGES CXX
)

# Produce GNU-style variables for installation, and default
# the installation directory to be local to the build. If you
# intend to install elsewhere, pass an explicit argument
# to CMAKE_INSTALL_PREFIX on the command line.
include(GNUInstallDirs)
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()

# Enforce the C++ standard, and disable extensions
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

# TODO: Do we want to default to shared libraries? Also see top-level
# TODO about whether we should build *both* if shared librares is ON
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Allow binaries in the build tree to be run meaningfully when BUILD_SHARED_LIBS=ON
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# Ensure that RPATHs required to reach external dependencies
# propagate into the instalation
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# For now, we require a `viam_rust_utils` in the root of the tree. The user
# is responsible for providing it. We then stage it into the installation.
#
# TODO: Error out if this glob doesn't match anything
# TODO: Consider downloading this library for the user if not present
file(GLOB viam_rust_utils_lib ${PROJECT_SOURCE_DIR}/libviam_rust_utils*.*)
install(
  FILES ${viam_rust_utils_lib}
  TYPE LIB
  COMPONENT runtime
)

add_subdirectory(src)

# TODO: Push examples under `src`?
add_subdirectory(examples)
