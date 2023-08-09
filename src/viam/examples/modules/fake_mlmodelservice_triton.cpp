// Copyright 2023 Viam Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dlfcn.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <chrono>
#include <thread>

#include "vtriton.hpp"

//#include <viam/sdk/spatialmath/geometry.hpp>

namespace {

constexpr char service_name[] = "fake_mlmodelservice_triton";

}  // namespace
int main() {
    vtriton::the_shim.ApiVersion = &TRITONSERVER_ApiVersion;
    vtriton::the_shim.ErrorCodeString = &TRITONSERVER_ErrorCodeString;
    vtriton::the_shim.ErrorDelete = &TRITONSERVER_ErrorDelete;
    vtriton::the_shim.ErrorMessage = &TRITONSERVER_ErrorMessage;
    vtriton::the_shim.ErrorNew = &TRITONSERVER_ErrorNew;
    vtriton::the_shim.ServerOptionsDelete = &TRITONSERVER_ServerOptionsDelete;
    vtriton::the_shim.ServerOptionsNew = &TRITONSERVER_ServerOptionsNew;
    vtriton::the_shim.ServerDelete = &TRITONSERVER_ServerDelete;
    vtriton::the_shim.ServerNew = &TRITONSERVER_ServerNew;
    vtriton::the_shim.ServerOptionsSetBackendDirectory = &TRITONSERVER_ServerOptionsSetBackendDirectory;
    vtriton::the_shim.ServerOptionsSetLogVerbose = &TRITONSERVER_ServerOptionsSetLogVerbose;
    vtriton::the_shim.ServerOptionsSetMinSupportedComputeCapability =
        &TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability;
    vtriton::the_shim.ServerOptionsSetModelRepositoryPath = &TRITONSERVER_ServerOptionsSetModelRepositoryPath;
    vtriton::the_shim.ServerOptionsSetStrictModelConfig = &TRITONSERVER_ServerOptionsSetStrictModelConfig;

    auto lfmtl = dlmopen(LM_ID_NEWLM, "libfake_mlmodelservice_triton_lib.so", RTLD_NOW);
    if (!lfmtl) {
        const auto err = dlerror();
        std::ostringstream buffer;
        buffer << service_name << ": Failed to open libfake_mlmodelservice_triton_lib.so Library: " << err;
        throw std::runtime_error(buffer.str());
    }

    void* lfmtl_serve = dlsym(lfmtl, "libfake_mlmodelservice_triton_lib_serve");
    if (!lfmtl_serve) {
        const auto err = dlerror();
        std::ostringstream buffer;
        buffer << service_name << ": Failed to find entry point in lfmtl libray: " << err;
        throw std::runtime_error(buffer.str());
    }

    std::cout << "XXX ACM OUTSIDE " << ((void *)(&vtriton::the_shim)) << " " << ((void *)(&vtriton::the_shim.ApiVersion));
    return reinterpret_cast<int(*)(vtriton::shim*)>(lfmtl_serve)(&vtriton::the_shim);
}
