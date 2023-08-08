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

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <chrono>
#include <thread>

#include <triton/core/tritonserver.h>

//#include <viam/sdk/spatialmath/geometry.hpp>

namespace {

constexpr char service_name[] = "example_mlmodelservice_triton";

// A namespace to bind unique_ptr and shared_ptr to the triton types in handy ways. Also
// provides other helpers specialized to interacting with the triton API, like vtriton::call.
namespace vtriton {

// Declare this here so we can use it, but the implementation relies on subsequent specializaitons.
template <typename Stdex = std::runtime_error, typename... Args>
[[gnu::warn_unused_result]] constexpr auto call(TRITONSERVER_Error* (*fn)(Args... args)) noexcept;

template <typename T>
struct lifecycle_traits;

template <typename T>
struct traits {
    using lifecycle = lifecycle_traits<T>;

    struct deleter_type {
        void operator()(T* t) const {
            lifecycle::dtor(t);
        }
    };

    using unique_ptr = std::unique_ptr<typename lifecycle::value_type, deleter_type>;
};

template <typename T>
using unique_ptr = typename traits<T>::unique_ptr;

template <typename T, class... Args>
unique_ptr<T> make_unique(Args&&... args) {
    using lifecycle = typename traits<T>::lifecycle;
    using unique_ptr = unique_ptr<T>;
    return unique_ptr(lifecycle::ctor(std::forward<Args>(args)...));
}

template <typename T>
unique_ptr<T> take_unique(T* t) {
    using lifecycle = typename traits<T>::lifecycle;
    using unique_ptr = unique_ptr<T>;
    return unique_ptr(t);
}

template <typename T, class... Args>
std::shared_ptr<T> make_shared(Args&&... args) {
    using lifecycle = typename traits<T>::lifecycle;
    return std::shared_ptr<T>(lifecycle::ctor(std::forward<Args>(args)...), lifecycle::dtor);
}

template <>
struct lifecycle_traits<TRITONSERVER_Error> {
    using value_type = TRITONSERVER_Error;
    static constexpr const auto ctor = TRITONSERVER_ErrorNew;
    static constexpr const auto dtor = TRITONSERVER_ErrorDelete;
};

template <>
struct lifecycle_traits<TRITONSERVER_ServerOptions> {
    using value_type = TRITONSERVER_ServerOptions;
    template <class... Args>
    static auto ctor(Args&&... args) {
        TRITONSERVER_ServerOptions* opts = nullptr;
        call(TRITONSERVER_ServerOptionsNew)(&opts, std::forward<Args>(args)...);
        return opts;
    };
    static constexpr const auto dtor = TRITONSERVER_ServerOptionsDelete;
};

template <>
struct lifecycle_traits<TRITONSERVER_Server> {
    using value_type = TRITONSERVER_Server;
    template <class... Args>
    static auto ctor(Args&&... args) {
        TRITONSERVER_Server* server = nullptr;
        call(TRITONSERVER_ServerNew)(&server, std::forward<Args>(args)...);
        return server;
    };
    static constexpr const auto dtor = TRITONSERVER_ServerDelete;
};

template <typename Stdex = std::runtime_error, typename... Args>
[[gnu::warn_unused_result]] constexpr auto call(TRITONSERVER_Error* (*fn)(Args... args)) noexcept {
    // NOTE: The lack of perfect forwarding here is deliberate. The Triton API is in C. We want
    // ordinary conversions (like std::string to const char*) to apply.
    //
    // TODO: Lies?
    return [=](Args... args) {
        const auto error = take_unique(fn(args...));
        if (error) {
            std::ostringstream buffer;
            buffer << service_name
                   << ": Triton Server Error: " << TRITONSERVER_ErrorCodeString(error.get())
                   << " - " << TRITONSERVER_ErrorMessage(error.get());
            throw Stdex(buffer.str());
        }
    };
}

}  // namespace vtriton

}  // namespace
int main() {
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

    vtriton::call(TRITONSERVER_ServerOptionsSetModelRepositoryPath)(server_options.get(),
                                                                    model_repo_path.c_str());

    vtriton::call(TRITONSERVER_ServerOptionsSetBackendDirectory)(server_options.get(),
                                                                 backend_directory.c_str());

    // TODO: Parameterize?
    vtriton::call(TRITONSERVER_ServerOptionsSetLogVerbose)(server_options.get(), 1);

    // Needed so we can load a tensorflow model without a config file
    // TODO: Maybe?
    vtriton::call(TRITONSERVER_ServerOptionsSetStrictModelConfig)(server_options.get(), false);

    // Per https://developer.nvidia.com/cuda-gpus, 5.3 is the lowest
    // value for all of the Jetson Line.
    //
    // TODO: Does setting this low constrain our GPU utilization in ways
    // that we don't like?
    vtriton::call(TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability)(server_options.get(),
                                                                              8.7);

    std::cout << "XXX ACM constructing server" << std::endl;
    auto server = vtriton::make_unique<TRITONSERVER_Server>(server_options.get());
    std::cout << "XXX ACM constructed server" << std::endl;

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(2000s);

    //viam::sdk::GeometryConfig config;
    
    return 0;
}
