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

# TODO: Consider removing this example entirely.

find_package(gRPC CONFIG REQUIRED)

add_executable(example_echo
  ${PROJECT_SOURCE_DIR}/src/gen/google/api/http.pb.cc
  ${PROJECT_SOURCE_DIR}/src/gen/google/api/annotations.pb.cc
  ${PROJECT_SOURCE_DIR}/src/gen/proto/rpc/examples/echo/v1/echo.pb.cc
  ${PROJECT_SOURCE_DIR}/src/gen/proto/rpc/examples/echo/v1/echo.grpc.pb.cc
  example_echo.cc
)

target_include_directories(example_echo
  PUBLIC ${PROJECT_SOURCE_DIR}/src/gen
)

target_link_libraries(example_echo
  ${viam_rust_utils_lib}
  gRPC::grpc++
)

install(
  TARGETS example_echo
  COMPONENT examples
)
