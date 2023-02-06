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

# TODO: DRY the two lists here. At least one file here
# must be in common with the list of outputs for invoking `buf generate`
# in order to create the dependency edge.

add_executable(example_echo
  ${PROTO_GEN_DIR}/gen/google/api/http.pb.cc
  ${PROTO_GEN_DIR}/gen/google/api/annotations.pb.cc
  ${PROTO_GEN_DIR}/gen/proto/rpc/examples/echo/v1/echo.pb.cc
  ${PROTO_GEN_DIR}/gen/proto/rpc/examples/echo/v1/echo.grpc.pb.cc
  example_echo.cc
)

# NOTE: Unlike the `dial` example which obtains the headers paths
# automatically by virtue of linking to `libviamcpp`, here we must
# explicitly set the include directories.
target_include_directories(example_echo
  PUBLIC ${PROJECT_SOURCE_DIR}/src
  PUBLIC ${PROTO_GEN_DIR}/gen
)

# Similarly, we can't pick up the public grpc or viam_rust_utils
# dependencis from viamcpp, so we need to be explicit about them.
target_link_libraries(example_echo
  gRPC::grpc++
  ${viam_rust_utils_lib}
)

install(
  TARGETS example_echo
  COMPONENT examples
)
