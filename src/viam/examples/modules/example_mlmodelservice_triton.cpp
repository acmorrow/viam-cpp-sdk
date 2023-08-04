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

#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <signal.h>
#include <sstream>
#include <stdexcept>

#include <grpcpp/channel.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <triton/core/tritonserver.h>

#include <viam/sdk/components/component.hpp>
#include <viam/sdk/config/resource.hpp>
#include <viam/sdk/module/service.hpp>
#include <viam/sdk/registry/registry.hpp>
#include <viam/sdk/rpc/server.hpp>
#include <viam/sdk/services/mlmodel/mlmodel.hpp>
#include <viam/sdk/services/mlmodel/server.hpp>

namespace {

namespace vsdk = ::viam::sdk;
constexpr char service_name[] = "example_mlmodelservice_triton";

// An example MLModelService instance which runs many models via the
// NVidia Triton Server API.x
//
// Configuration requires the following parameters:
//    TODO
class MLModelServiceTriton : public vsdk::MLModelService {
   public:
    explicit MLModelServiceTriton(vsdk::Dependencies dependencies,
                                  vsdk::ResourceConfig configuration)
        : MLModelService(configuration.name()),
          state_(reconfigure_(std::move(dependencies), std::move(configuration))) {
        std::cout << "XXX ACM MLModelServiceTriton: instantiated as '" << this->name() << "'" << std::endl;
    }

    ~MLModelServiceTriton() final {
        // All invocations arrive via gRPC, so we know we are idle
        // here. It should be safe to tear down all state
        // automatically without needing to wait for anything more to
        // drain.
    }

    grpc::StatusCode stop(vsdk::AttributeMap extra) noexcept final {
        return stop();
    }

    /// @brief Stops a resource from running.
    grpc::StatusCode stop() noexcept final {
        using std::swap;
        try {
            std::lock_guard<std::mutex> lock(state_lock_);
            if (!stopped_) {
                stopped_ = true;
                std::shared_ptr<state> state;
                swap(state_, state);
                state_ready_.notify_all();
            }
        } catch (...) {
        }
        return grpc::StatusCode::OK;
    }

    void reconfigure(vsdk::Dependencies dependencies, vsdk::ResourceConfig configuration) final
        try {
        // Care needs to be taken during reconfiguration. The
        // framework does not offer protection against invocation
        // during reconfiguration. Keep all state in a shared_ptr
        // managed block, and allow client invocations to act against
        // current state while a new configuration is built, then swap
        // in the new state. State which is in use by existing
        // invocations will remain valid until the clients drain. If
        // reconfiguration fails, the component will `stop`.

        // Swap out the state_ member with nullptr. Existing
        // invocations will continue to operate against the state they
        // hold, and new invocations will block on the state becoming
        // populated.
        using std::swap;
        std::shared_ptr<state> state;
        {
            // Wait until we have a state in play, then take
            // ownership, so that we don't race with other
            // reconfigurations and so other invocations wait on a new
            // state.
            std::unique_lock<std::mutex> lock(state_lock_);
            state_ready_.wait(lock, [this]() { return (state_ != nullptr) && !stopped_; });
            check_stopped_inlock_();
            swap(state_, state);
        }

        state = reconfigure_(std::move(dependencies), std::move(configuration));

        // Reconfiguration worked: put the state in under the lock,
        // release the lock, and then notify any callers waiting on
        // reconfiguration to complete.
        {
            std::lock_guard<std::mutex> lock(state_lock_);
            check_stopped_inlock_();
            swap(state_, state);
        }
        state_ready_.notify_all();
    } catch (...) {
        // If reconfiguration fails for any reason, become stopped and rethrow.
        stop();
        throw;
    }

    std::shared_ptr<named_tensor_views> infer(const named_tensor_views& inputs) final {
        std::cout << "XXX ACM MLModelServiceTriton: recieved `infer` invocation" << std::endl;
        const auto state = lease_state_();

        static constexpr std::array<float, 100> location_data = {0.1, 0.1, 0.75, 0.75};
        static constexpr std::array<float, 25> category_data = {0};
        static constexpr std::array<float, 25> score_data = {.99};
        static constexpr std::array<float, 1> num_dets_data = {25};

        auto location_tensor =
            make_tensor_view(location_data.data(), location_data.size(), {1, 25, 4});

        auto category_tensor =
            make_tensor_view(category_data.data(), category_data.size(), {1, 25});

        auto score_tensor = make_tensor_view(score_data.data(), score_data.size(), {1, 25});

        auto num_dets_tensor = make_tensor_view(num_dets_data.data(), num_dets_data.size(), {1});

        using namespace std::literals::string_literals;
        named_tensor_views tensors{{"location"s, std::move(location_tensor)},
                                   {"category"s, std::move(category_tensor)},
                                   {"score"s, std::move(score_tensor)},
                                   {"n_detections"s, std::move(num_dets_tensor)}};

        return std::make_shared<named_tensor_views>(std::move(tensors));
        return {};
    }

    struct metadata metadata() final {
        std::cout << "XXX ACM MLModelServiceTriton: recieved `metadata` invocation" << std::endl;
        // Just return a copy of our metadata from leased state.
        const auto state = lease_state_();
                // This metadata is modelled on the results obtained from
        // invoking `Metadata` on a instance of tflite_cpu configured
        // per the instructions and data at
        // https://github.com/viamrobotics/vision-service-examples/tree/aa4195485754151fccbfd61fbe8bed63db7f300f

        return {// `name`
                "C++ SDK Example MLModel - Faking EfficientDet Lite0 V1",

                // `type`
                "fake_tflite_detector",

                // `description`
                "Identify which of a known set of objects might be present and provide "
                "information "
                "about their positions within the given image or a video stream.",

                // `inputs`
                {
                    {// `name`
                     "image",

                     // `description`
                     "Input image to be detected.",

                     // `data_type`
                     tensor_info::data_types::k_uint8,

                     // `shape`
                     {1, 320, 320, 3},

                     // `associated_files`
                     {},

                     // `extra`
                     {}},
                },

                // `outputs`
                {{// `name`
                  "location",

                  // `description`
                  "The locations of the detected boxes.",

                  // `data_type`
                  tensor_info::data_types::k_float32,

                  // `shape`
                  {1, 25, 4},

                  // `associated_files`
                  {},

                  // `extra`
                  std::make_shared<vsdk::AttributeMap::element_type>(
                      std::initializer_list<vsdk::AttributeMap::element_type::value_type>{
                          {"labels", std::make_shared<vsdk::ProtoType>("/example/labels.txt")}})},

                 {// `name`
                  "category",

                  // `description`
                  "The categories of the detected boxes.",

                  // `data_type`
                  tensor_info::data_types::k_float32,

                  // `shape`
                  {1, 25},

                  // `associated files`
                  {{
                      // `name`
                      "labelmap.txt",

                      // `description`
                      "Label of objects that this model can recognize.",

                      MLModelService::tensor_info::file::k_label_type_tensor_value,
                  }},

                  // `extra`
                  {}},

                 {// `name`
                  "score",

                  // `description`
                  "The scores of the detected boxes.",

                  // `data_type`
                  tensor_info::data_types::k_float32,

                  // `shape`
                  {1, 25},

                  // `associated_files`
                  {},

                  // `extra`
                  {}},

                 {// `name`
                  "n_detections",

                  // `description`
                  "The number of the detected boxes.",

                  // `data_type`
                  tensor_info::data_types::k_float32,

                  // `shape`
                  {1},

                  // `associated_files`
                  {},

                  // `extra`
                  {}}}};
    }

   private:
    struct state;

    void check_stopped_inlock_() const {
        if (stopped_) {
            std::ostringstream buffer;
            buffer << service_name << ": service is stopped: ";
            throw std::runtime_error(buffer.str());
        }
    }

    std::shared_ptr<state> lease_state_() {
        // Wait for our state to be valid or stopped and then obtain a
        // shared_ptr to state if valid, incrementing the refcount, or
        // throws if the service is stopped. We don't need to deal
        // with interruption or shutdown because the gRPC layer will
        // drain requests during shutdown, so it shouldn't be possible
        // for callers to get stuck here.
        std::unique_lock<std::mutex> lock(state_lock_);
        state_ready_.wait(lock, [this]() { return (state_ != nullptr) && !stopped_; });
        check_stopped_inlock_();
        return state_;
    }

    static std::shared_ptr<state> reconfigure_(vsdk::Dependencies dependencies,
                                               vsdk::ResourceConfig configuration) {
        //return std::make_shared<state>();
        return std::make_shared<state>(std::move(dependencies), std::move(configuration));
    }

    // All of the meaningful internal state of the service is held in
    // a separate state object so we can keep our current state alive
    // while building a new one during reconfiguration, and then
    // atomically swap it in on success. Existing invocations will
    // continue to work against the old state, and new invocations
    // will pick up the new state.
    struct state {
        explicit state(vsdk::Dependencies dependencies, vsdk::ResourceConfig configuration)
            : dependencies(std::move(dependencies)), configuration(std::move(configuration)) {}

        // The dependencies and configuration we were given at
        // construction / reconfiguration.
        vsdk::Dependencies dependencies;
        vsdk::ResourceConfig configuration;
    };

    // The mutex and condition variable needed to track our state
    // across concurrent reconfiguration and invocation.
    std::mutex state_lock_;
    std::condition_variable state_ready_;
    std::shared_ptr<state> state_;
    bool stopped_ = false;
};

int serve(const std::string& socket_path) try {
    // Block the signals we intend to wait for synchronously.
    sigset_t sigset;
    sigemptyset(&sigset);
    sigaddset(&sigset, SIGINT);
    sigaddset(&sigset, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &sigset, NULL);

    // Create a new model registration for the service.
    auto module_registration = std::make_shared<vsdk::ModelRegistration>(
        // TODO(RSDK-2417): This field is deprecated, just provide some
        // quasi-relevant string value here for now.
        vsdk::ResourceType{"MLModelServiceTritonModule"},

        // Identify that this resource offers the MLModelService API
        vsdk::MLModelService::static_api(),

        // Declare a model triple for this service.
        vsdk::Model{"viam", "mlmodelservice", "example_mlmodelservice_triton"},

        // Define the factory for instances of the resource.
        [](vsdk::Dependencies deps, vsdk::ResourceConfig config) {
            return std::make_shared<MLModelServiceTriton>(std::move(deps), std::move(config));
        });

    // Register the newly created registration with the Registry.
    vsdk::Registry::register_model(module_registration);

    // Construct the module service and tell it where to place the socket path.
    auto module_service = std::make_shared<vsdk::ModuleService_>(socket_path);

    // Construct a new Server object.
    auto server = std::make_shared<vsdk::Server>();

    // Add the server as providing the API and model declared in the
    // registration.
    module_service->add_model_from_registry(
        server, module_registration->api(), module_registration->model());

    // Start the module service.
    module_service->start(server);

    // Create a thread which will start the server, await one of the
    // blocked signals, and then gracefully shut down the server.
    std::thread server_thread([&server, &sigset]() {
        server->start();
        int sig = 0;
        auto result = sigwait(&sigset, &sig);
        server->shutdown();
    });

    // The main thread waits for the server thread to indicate that
    // the server shutdown has completed.
    server->wait();

    // Wait for the server thread to exit.
    server_thread.join();

    return EXIT_SUCCESS;
} catch (const std::exception& ex) {
    std::cout << "ERROR: A std::exception was thrown from `serve`: " << ex.what() << std::endl;
    return EXIT_FAILURE;
} catch (...) {
    std::cout << "ERROR: An unknown exception was thrown from `serve`" << std::endl;
    return EXIT_FAILURE;
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::string usage = "usage: example_mlmodelservice_triton /path/to/unix/socket";

    std::uint32_t api_version_major, api_version_minor;
    TRITONSERVER_ApiVersion(&api_version_major, &api_version_minor);
    std::cout << "TRITONSERVER_API_VERSION_MAJOR: " << TRITONSERVER_API_VERSION_MAJOR << std::endl;
    std::cout << "TRITONSERVER_API_VERSION_MINOR: " << TRITONSERVER_API_VERSION_MINOR << std::endl;
    std::cout << "api_version_major:              " << api_version_major << std::endl;
    std::cout << "api_version_minor:              " << api_version_minor << std::endl;

    if (argc < 2) {
        std::cout << "ERROR: insufficient arguments\n";
        std::cout << usage << "\n";
        return EXIT_FAILURE;
    }

    return serve(argv[1]);
}
