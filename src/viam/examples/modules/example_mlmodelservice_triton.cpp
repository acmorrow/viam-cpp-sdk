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

// An example MLModelService instance which runs many models via the
// NVidia Triton Server API.x
//
// Configuration requires the following parameters:
//    TODO
class MLModelServiceTriton : public vsdk::MLModelService {
   public:
    explicit MLModelServiceTriton(vsdk::Dependencies dependencies,
                                  vsdk::ResourceConfig configuration) try
        : MLModelService(configuration.name()),
          state_(reconfigure_(std::move(dependencies), std::move(configuration))) {
        std::cout << "XXX ACM MLModelServiceTriton: instantiated as '" << this->name() << "'"
                  << std::endl;
    } catch (...) {
        std::cout << "XXX ACM MLModelServiceTriton::MLModelServiceTriton CTOR XCP" << std::endl;
        try {
            throw;

        } catch (const std::exception& xcp) {
            std::cout
                << "XXX ACM MLModelServiceTriton::MLModelServiceTriton CTOR XCP (std::exception) "
                << xcp.what() << std::endl;
            throw;
        }
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
        // return std::make_shared<state>();
        auto state =
            std::make_shared<struct state>(std::move(dependencies), std::move(configuration));

        // Validate that our dependencies (if any - we don't actually
        // expect any for this service) exist. If we did have
        // Dependencies this is where we would have an opportunity to
        // downcast them to the right thing and store them in our
        // state so we could use them as needed.
        //
        // TODO(RSDK-3601): Validating that dependencies are present
        // should be handled by the ModuleService automatically,
        // rather than requiring each component to validate the
        // presence of dependencies.
        for (const auto& kv : state->dependencies) {
            if (!kv.second) {
                std::ostringstream buffer;
                buffer << service_name << ": Dependency "
                       << "`" << kv.first.to_string() << "` was not found during (re)configuration";
                throw std::invalid_argument(buffer.str());
            }
        }

        const auto& attributes = state->configuration.attributes();

        // Pull the model repository path out of the configuration.
        auto model_repo_path = attributes->find("model_repository_path");
        if (model_repo_path == attributes->end()) {
            std::ostringstream buffer;
            buffer << service_name
                   << ": Required parameter `model_repository_path` not found in configuration";
            throw std::invalid_argument(buffer.str());
        }

        auto* const model_repo_path_string = model_repo_path->second->get<std::string>();
        if (!model_repo_path_string || model_repo_path_string->empty()) {
            std::ostringstream buffer;
            buffer << service_name
                   << ": Required non-empty string parameter `model_repository_path` is either not "
                      "a string "
                      "or is an empty string";
            throw std::invalid_argument(buffer.str());
        }
        // TODO(acm): Maybe validate path is a readable directory, if triton doesn't do so.
        state->model_repo_path = std::move(*model_repo_path_string);

        // Pull the backend directory out of the configuration.
        //
        // TODO: Does this really belong in the config? Or should it be part of the docker
        // setup?
        auto backend_directory = attributes->find("backend_directory");
        if (backend_directory == attributes->end()) {
            std::ostringstream buffer;
            buffer << service_name
                   << ": Required parameter `backend_directory` not found in configuration";
            throw std::invalid_argument(buffer.str());
        }

        auto* const backend_directory_string = backend_directory->second->get<std::string>();
        if (!backend_directory_string || backend_directory_string->empty()) {
            std::ostringstream buffer;
            buffer << service_name
                   << ": Required non-empty string parameter `backend_directory` is either not a "
                      "string "
                      "or is an empty string";
            throw std::invalid_argument(buffer.str());
        }
        // TODO(acm): Maybe validate path is a readable directory, if triton doesn't do so.
        state->backend_directory = std::move(*backend_directory_string);

        // Pull the model name out of the configuration.
        auto model_name = attributes->find("model_name");
        if (model_name == attributes->end()) {
            std::ostringstream buffer;
            buffer << service_name
                   << ": Required parameter `model_name` not found in configuration";
            throw std::invalid_argument(buffer.str());
        }

        auto* const model_name_string = model_name->second->get<std::string>();
        if (!model_name_string || model_name_string->empty()) {
            std::ostringstream buffer;
            buffer << service_name
                   << ": Required non-empty string parameter `model_name` is either not a "
                      "string "
                      "or is an empty string";
            throw std::invalid_argument(buffer.str());
        }
        state->model_name = std::move(*model_name_string);

        // Process any tensor name remappings provided in the config.
        auto remappings = attributes->find("tensor_name_remappings");
        if (remappings != attributes->end()) {
            const auto remappings_attributes = remappings->second->get<vsdk::AttributeMap>();
            if (!remappings_attributes) {
                std::ostringstream buffer;
                buffer << service_name
                       << ": Optional parameter `tensor_name_remappings` must be a dictionary";
                throw std::invalid_argument(buffer.str());
            }

            const auto populate_remappings = [](const vsdk::ProtoType& source, auto& target) {
                const auto source_attributes = source.get<vsdk::AttributeMap>();
                if (!source_attributes) {
                    std::ostringstream buffer;
                    buffer << service_name
                           << ": Fields `inputs` and `outputs` of `tensor_name_remappings` must be "
                              "dictionaries";
                    throw std::invalid_argument(buffer.str());
                }
                for (const auto& kv : *source_attributes) {
                    const auto& k = kv.first;
                    const auto* const kv_string = kv.second->get<std::string>();
                    if (!kv_string) {
                        std::ostringstream buffer;
                        buffer
                            << service_name
                            << ": Fields `inputs` and `outputs` of `tensor_name_remappings` must "
                               "be dictionaries with string values";
                        throw std::invalid_argument(buffer.str());
                    }
                    target[kv.first] = *kv_string;
                }
            };

            const auto inputs_where = remappings_attributes->find("inputs");
            if (inputs_where != remappings_attributes->end()) {
                populate_remappings(*inputs_where->second, state->input_name_remappings);
            }
            const auto outputs_where = remappings_attributes->find("outputs");
            if (outputs_where != remappings_attributes->end()) {
                populate_remappings(*outputs_where->second, state->output_name_remappings);
            }
        }

        auto server_options = vtriton::make_unique<TRITONSERVER_ServerOptions>();

        vtriton::call(TRITONSERVER_ServerOptionsSetModelRepositoryPath)(
            server_options.get(), state->model_repo_path.c_str());

        vtriton::call(TRITONSERVER_ServerOptionsSetBackendDirectory)(
            server_options.get(), state->backend_directory.c_str());

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
        vtriton::call(TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability)(
            server_options.get(), 8.7);

        std::cout << "XXX ACM constructing server" << std::endl;
        state->server = vtriton::make_unique<TRITONSERVER_Server>(server_options.get());
        std::cout << "XXX ACM constructed server" << std::endl;

        return state;
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


        ~state() {
            std::cout << "XXX ACM Destroying State" << std::endl;
        }

        // The dependencies and configuration we were given at
        // construction / reconfiguration.
        vsdk::Dependencies dependencies;
        vsdk::ResourceConfig configuration;

        // The path to the model repository. The provided directory must
        // meet the layout requirements for a triton model repository. See
        //
        // https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md
        std::string model_repo_path;

        // The path to the backend directory containing execution backends.
        std::string backend_directory;

        // The name of the specific model that this instance of the
        // triton service will bind to.
        std::string model_name;

        // Tensor renamings as extracted from our configuration. The
        // keys are the names of the tensors per the model, the values
        // are the names of the tensors clients expect to see / use
        // (e.g. a vision service component expecting a tensor named
        // `image`).
        std::unordered_map<std::string, std::string> input_name_remappings;
        std::unordered_map<std::string, std::string> output_name_remappings;

        vtriton::unique_ptr<TRITONSERVER_Server> server;
    };

    // The mutex and condition variable needed to track our state
    // across concurrent reconfiguration and invocation.
    std::mutex state_lock_;
    std::condition_variable state_ready_;
    std::shared_ptr<state> state_;
    bool stopped_ = false;
};

int serve(const std::string& socket_path) noexcept try {
    // Block the signals we intend to wait for synchronously.
    sigset_t sigset;
    sigemptyset(&sigset);
    sigaddset(&sigset, SIGINT);
    sigaddset(&sigset, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &sigset, NULL);

    // Validate that the version of the triton server that we are
    // running against is sufficient w.r..t the version we were built
    // against.
    std::uint32_t triton_version_major;
    std::uint32_t triton_version_minor;
    vtriton::call(TRITONSERVER_ApiVersion)(&triton_version_major, &triton_version_minor);

    if ((TRITONSERVER_API_VERSION_MAJOR != triton_version_major) ||
        (TRITONSERVER_API_VERSION_MINOR > triton_version_minor)) {
        std::ostringstream buffer;
        buffer << service_name << ": Triton server API version mismatch: need "
               << TRITONSERVER_API_VERSION_MAJOR << "." << TRITONSERVER_API_VERSION_MINOR
               << " but have " << triton_version_major << "." << triton_version_minor << ".";
        throw std::domain_error(buffer.str());
    }
    std::cout << service_name << ": Running Triton API " << triton_version_major << "."
              << triton_version_minor << std::endl;

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

    if (argc < 2) {
        std::cout << "ERROR: insufficient arguments\n";
        std::cout << usage << "\n";
        return EXIT_FAILURE;
    }

    return serve(argv[1]);
}
