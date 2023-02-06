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

find_package(gRPC CONFIG REQUIRED)
find_package(Boost REQUIRED
  COMPONENTS log
)

add_library(viamcpp
  gen/common/v1/common.grpc.pb.cc
  gen/common/v1/common.pb.cc
  gen/google/api/annotations.pb.cc
  gen/google/api/http.pb.cc
  gen/robot/v1/robot.pb.cc
  gen/robot/v1/robot.grpc.pb.cc
  common/proto_type.cpp
  common/utils.cpp
  components/component_base.cpp
  components/resource_manager.cpp
  registry/registry.cpp
  robot/client.cpp
  rpc/dial.cpp
)

# TODO: This doesn't install with the right path
# set_target_properties(viamcpp
#   PROPERTIES PUBLIC_HEADER "robot/client.hpp"
# )

target_include_directories(viamcpp
  PUBLIC ${PROJECT_SOURCE_DIR}/src
  PUBLIC ${PROJECT_SOURCE_DIR}/src/gen
)

target_link_libraries(viamcpp
  PUBLIC gRPC::grpc++
  PRIVATE ${Boost_LOG_LIBRARY}
  PRIVATE ${viam_rust_utils_lib}
)

install(
  TARGETS viamcpp
  LIBRARY COMPONENT runtime
  PUBLIC_HEADER COMPONENT dev
)
