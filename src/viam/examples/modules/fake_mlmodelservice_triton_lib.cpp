#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "vtriton.hpp"

#include <viam/sdk/spatialmath/geometry.hpp>

namespace {

constexpr char service_name[] = "fake_mlmodelservice_triton";

}  // namespace

extern "C" int libfake_mlmodelservice_triton_lib_serve(vtriton::shim* shim) {
    viam::sdk::GeometryConfig config;

    vtriton::the_shim = *shim;
    std::cout << "XXX ACM in fmtl: " << ((void *)shim) << " " << ((void *)(vtriton::the_shim.ApiVersion)) << std::endl;
    // Validate that the version of the triton server that we are
    // running against is sufficient w.r..t the version we were built
    // against.
    std::uint32_t triton_version_major;
    std::uint32_t triton_version_minor;
    vtriton::call(vtriton::the_shim.ApiVersion)(&triton_version_major, &triton_version_minor);

    if ((TRITONSERVER_API_VERSION_MAJOR != triton_version_major) ||
        (TRITONSERVER_API_VERSION_MINOR > triton_version_minor)) {
        std::ostringstream buffer;
        buffer << service_name << ": Triton server API version mismatch: need "
               << TRITONSERVER_API_VERSION_MAJOR << "." << TRITONSERVER_API_VERSION_MINOR
               << " but have " << triton_version_major << "." << triton_version_minor << ".";
        throw std::domain_error(buffer.str());
    }
    std::cout << service_name << ": Running Triton API (via dlopen)" << triton_version_major << "."
              << triton_version_minor << std::endl;

    // Pull the model repository path out of the configuration.
    std::string model_repo_path = "/home/andrewmorrow/.viam/triton";

    // Pull the backend directory out of the configuration.
    //
    // TODO: Does this really belong in the config? Or should it be part of the docker
    // setup?
    std::string backend_directory = "/opt/tritonserver/backends";

    // Pull the model name out of the configuration.
    std::string model_name = "efficientdet-lite0-detection";

    auto server_options = vtriton::make_unique<TRITONSERVER_ServerOptions>();

    vtriton::call(vtriton::the_shim.ServerOptionsSetModelRepositoryPath)(server_options.get(),
                                                                    model_repo_path.c_str());

    vtriton::call(vtriton::the_shim.ServerOptionsSetBackendDirectory)(server_options.get(),
                                                                 backend_directory.c_str());

    // TODO: Parameterize?
    vtriton::call(vtriton::the_shim.ServerOptionsSetLogVerbose)(server_options.get(), 1);

    // Needed so we can load a tensorflow model without a config file
    // TODO: Maybe?
    vtriton::call(vtriton::the_shim.ServerOptionsSetStrictModelConfig)(server_options.get(), false);

    // Per https://developer.nvidia.com/cuda-gpus, 5.3 is the lowest
    // value for all of the Jetson Line.
    //
    // TODO: Does setting this low constrain our GPU utilization in ways
    // that we don't like?
    vtriton::call(vtriton::the_shim.ServerOptionsSetMinSupportedComputeCapability)(server_options.get(),
                                                                              8.7);

    std::cout << "XXX ACM constructing server" << std::endl;
    auto server = vtriton::make_unique<TRITONSERVER_Server>(server_options.get());
    std::cout << "XXX ACM constructed server" << std::endl;

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(2000s);



    return 0;

}
