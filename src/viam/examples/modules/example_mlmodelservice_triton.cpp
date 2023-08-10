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

#include <dlfcn.h>

#include <iostream>
#include <string>

#include "vtriton.hpp"

namespace {
constexpr char service_name[] = "example_mlmodelservice_triton";
const std::string usage = "usage: example_mlmodelservice_triton /path/to/unix/socket";
}  // namespace

int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cout << service_name << "ERROR: insufficient arguments\n";
        std::cout << usage << std::endl;;
        return EXIT_FAILURE;
    }

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

    auto lemtl = dlopen("libexample_mlmodelservice_triton_server.so", RTLD_NOW);
    if (!lemtl) {
        const auto err = dlerror();
        std::cout << service_name << ": Failed to open libexample_mlmodelservice_triton_server.so Library: " << err;
        return EXIT_FAILURE;
    }

    const auto lemtl_serve = dlsym(lemtl, "example_mlmodelservice_triton_serve");
    if (!lemtl_serve) {
        const auto err = dlerror();
        std::cout << service_name << ": Failed to find entry point in server libray: " << err;
        return EXIT_FAILURE;
    }

    std::cout << "XXX ACM OUTSIDE " << ((void *)(&vtriton::the_shim)) << " " << ((void *)(&vtriton::the_shim.ApiVersion));
    return reinterpret_cast<int(*)(vtriton::shim*, const char*)>(lemtl_serve)(&vtriton::the_shim, argv[1]);
}
