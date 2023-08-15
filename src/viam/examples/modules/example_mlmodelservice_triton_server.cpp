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

#include <pthread.h>
#include <signal.h>

#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stack>
#include <stdexcept>

#include <grpcpp/channel.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <viam/sdk/components/component.hpp>
#include <viam/sdk/config/resource.hpp>
#include <viam/sdk/module/service.hpp>
#include <viam/sdk/registry/registry.hpp>
#include <viam/sdk/rpc/server.hpp>
#include <viam/sdk/services/mlmodel/mlmodel.hpp>
#include <viam/sdk/services/mlmodel/server.hpp>

#include "vtriton.hpp"

#include <cuda_runtime_api.h>

namespace {

namespace vsdk = ::viam::sdk;
constexpr char service_name[] = "example_mlmodelservice_triton";

static const auto cuda_deleter = [](void* ptr) {
    if (!ptr)
        return;

    cudaPointerAttributes cuda_attrs;
    auto cuda_error = cudaPointerGetAttributes(&cuda_attrs, ptr);
    // TODO: examine error and (throw?)
    // TODO: set device?
    switch (cuda_attrs.type) {
        case cudaMemoryTypeDevice: {
            cuda_error = cudaFree(ptr);
            break;
        }
        case cudaMemoryTypeHost: {
            cuda_error = cudaFreeHost(ptr);
            break;
        }
        default: {
            // TODO: abort?
        }
    }
    // TODO: examine error
};

// A namespace to bind unique_ptr and shared_ptr to the triton types in handy ways. Also
// provides other helpers specialized to interacting with the triton API, like vtriton::call.

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
        std::cerr << "XXX ACM MLModelServiceTriton::MLModelServiceTriton CTOR XCP" << std::endl;
        try {
            throw;

        } catch (const std::exception& xcp) {
            std::cerr
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

    std::shared_ptr<named_tensor_views> infer(const named_tensor_views& inputs) final try {
        std::cout << "XXX ACM MLModelServiceTriton: recieved `infer` invocation" << std::endl;
        const auto state = lease_state_();

        // TODO: Ensure that enough inputs were provided by comparing with metadata

        auto inference_request = get_inference_request_(state);

        // Attach inputs to the inference request
        std::stack<cuda_unique_ptr> cuda_allocations;
        for (const auto& kv : inputs) {
            const std::string* input_name = &kv.first;
            const auto where = state->input_name_remappings_reversed.find(*input_name);
            if (where != state->input_name_remappings_reversed.end()) {
                input_name = &where->second;
            }
            inference_request_input_visitor_ visitor(input_name,
                                                     inference_request.get(),
                                                     // TODO: config option?
                                                     TRITONSERVER_MEMORY_GPU,
                                                     // TODO: config option?
                                                     0);
            cuda_allocations.push(boost::apply_visitor(visitor, kv.second));
        }

        std::promise<TRITONSERVER_InferenceResponse*> inference_promise;
        auto inference_future = inference_promise.get_future();

        vtriton::call(vtriton::the_shim.InferenceRequestSetResponseCallback)(
            inference_request.get(),
            state->allocator.get(),
            state.get(),
            [](TRITONSERVER_InferenceResponse* response,
               const uint32_t flags,
               void* userp) noexcept {
                auto* promise =
                    reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
                promise->set_value(response);
            },
            &inference_promise);

        vtriton::call(vtriton::the_shim.ServerInferAsync)(
            state->server.get(), inference_request.release(), nullptr);

        auto result2 = inference_future.get();
        auto inference_response = vtriton::take_unique(result2);

        auto error = TRITONSERVER_InferenceResponseError(inference_response.get());
        if (error) {
            std::ostringstream buffer;
            buffer << ": Triton Server Inference Error: "
                   << vtriton::the_shim.ErrorCodeString(error) << " - "
                   << vtriton::the_shim.ErrorMessage(error);
            std::cerr << "XXX ACM Inference Failed!" << buffer.str() << std::endl;
            throw std::runtime_error(buffer.str());
        }
#if 0
        std::uint32_t num_outputs = 0;
        vtriton::call(vtriton::the_shim.InferenceResponseOutputCount)(inference_response.get(),
                                                                      &output_count);

        for (decltype(num_outputs) output_id = 0; output_id != num_outputs; ++output_id) {
            const char* output_cstr;
            TRITONSERVER_DataType type;
            const std::uint64_t* shape;
            std::uint64_t shape_size;
            const void* data;
            std::size_t data_bytes;
            TRITONSERVER_MemoryType memory_type;
            std::int64_t memory_type_id;

            vtriton::call(vtriton::the_shim.InferenceResponseOutput)(inference_response.get(),
                                                                     output_id,
                                                                     &name,
                                                                     &type,
                                                                     &shape,
                                                                     &shape_size,
                                                                     &data,
                                                                     &data_bytes,
                                                                     &memory_type,
                                                                     &memory_type_id,
                                                                     nullptr);

            if (!output_cstr) {
                // TODO: Log this? Return an error?
                continue;
            }

            std::string output(output_cstr);
            
        }

#else
        static constexpr std::array<float, 400>
            location_data = {0.1, 0.1, 0.75, 0.75};
        static constexpr std::array<float, 100> category_data = {0};
        static constexpr std::array<float, 100> score_data = {.99};
        static constexpr std::array<float, 1> num_dets_data = {100};

        auto location_tensor =
            make_tensor_view(location_data.data(), location_data.size(), {1, 100, 4});

        auto category_tensor =
            make_tensor_view(category_data.data(), category_data.size(), {1, 100});

        auto score_tensor = make_tensor_view(score_data.data(), score_data.size(), {1, 100});

        auto num_dets_tensor = make_tensor_view(num_dets_data.data(), num_dets_data.size(), {1, 1});

        using namespace std::literals::string_literals;
        named_tensor_views tensors{{"location"s, std::move(location_tensor)},
                                   {"category"s, std::move(category_tensor)},
                                   {"score"s, std::move(score_tensor)},
                                   {"n_detections"s, std::move(num_dets_tensor)}};

        auto result = std::make_shared<named_tensor_views>(std::move(tensors));
        return result;
#endif
    } catch (const std::exception& xcp) {
        std::cerr << "XXX ACM MLModelServiceTriton: Infer failed with exception: " << xcp.what()
                  << std::endl;
        throw;
    } catch (...) {
        std::cerr << "XXX ACM MLModelServiceTriton: Infer failed with an unknown exception"
                  << std::endl;
        throw;
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
                     {-1, 640, 480, 3},

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
                  {-1, 100, 4},

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
                  {-1, 100},

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
                  {-1, 100},

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
                  {-1, 1},

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

            const auto populate_remappings = [](const vsdk::ProtoType& source,
                                                auto& target,
                                                auto& inv_target) {
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
                    inv_target[*kv_string] = kv.first;
                }
            };

            const auto inputs_where = remappings_attributes->find("inputs");
            if (inputs_where != remappings_attributes->end()) {
                populate_remappings(*inputs_where->second,
                                    state->input_name_remappings,
                                    state->input_name_remappings_reversed);
            }
            const auto outputs_where = remappings_attributes->find("outputs");
            if (outputs_where != remappings_attributes->end()) {
                populate_remappings(*outputs_where->second,
                                    state->output_name_remappings,
                                    state->output_name_remappings_reversed);
            }
        }

        auto allocator = vtriton::make_unique<TRITONSERVER_ResponseAllocator>(
            allocate_response_, deallocate_response_, nullptr);

        auto server_options = vtriton::make_unique<TRITONSERVER_ServerOptions>();

        // TODO: We should probably pool servers based on repo path
        // and backend directory.
        vtriton::call(vtriton::the_shim.ServerOptionsSetModelRepositoryPath)(
            server_options.get(), state->model_repo_path.c_str());

        vtriton::call(vtriton::the_shim.ServerOptionsSetBackendDirectory)(
            server_options.get(), state->backend_directory.c_str());

        // TODO: Parameterize?
        vtriton::call(vtriton::the_shim.ServerOptionsSetLogVerbose)(server_options.get(), 0);

        // Needed so we can load a tensorflow model without a config file
        // TODO: Maybe?
        vtriton::call(vtriton::the_shim.ServerOptionsSetStrictModelConfig)(server_options.get(),
                                                                           false);

        // Per https://developer.nvidia.com/cuda-gpus, 5.3 is the lowest
        // value for all of the Jetson Line.
        //
        // TODO: Does setting this low constrain our GPU utilization in ways
        // that we don't like?
        vtriton::call(vtriton::the_shim.ServerOptionsSetMinSupportedComputeCapability)(
            server_options.get(), 8.7);

        auto server = vtriton::make_unique<TRITONSERVER_Server>(server_options.get());

        // TODO: These limits / timeouts should probably be configurable.
        bool result = false;
        for (size_t tries = 0; tries != 30; ++tries) {
            vtriton::call(vtriton::the_shim.ServerIsLive)(server.get(), &result);
            if (result) {
                vtriton::call(vtriton::the_shim.ServerIsReady)(server.get(), &result);
                if (result) {
                    std::cout << "XXX ACM Triton Server is live and ready" << std::endl;
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        if (!result) {
            std::cerr << "XXX ACM Triton Server did not become live and ready" << std::endl;
            throw std::runtime_error("Triton Server did not become live and ready within 30s");
        }

        TRITONSERVER_Message* model_metadata = nullptr;
        vtriton::call(vtriton::the_shim.ServerModelMetadata)(
            server.get(), state->model_name.c_str(), -1, &model_metadata);

        const char* json_base;
        size_t json_size;
        vtriton::call(vtriton::the_shim.MessageSerializeToJson)(
            model_metadata, &json_base, &json_size);
        std::string json_string(json_base, json_size);
        std::cout << "XXX ACM Model Metadata:" << json_string << std::endl;

        // TODO: Parse model metadata
        // TODO: Delete metadata Message

        state->allocator = std::move(allocator);
        state->server = std::move(server);

        return state;
    }

    static TRITONSERVER_Error* allocate_response_(TRITONSERVER_ResponseAllocator* allocator,
                                                  const char* tensor_name,
                                                  std::size_t byte_size,
                                                  TRITONSERVER_MemoryType memory_type,
                                                  std::int64_t memory_type_id,
                                                  void* userp,
                                                  void** buffer,
                                                  void** buffer_userp,
                                                  TRITONSERVER_MemoryType* actual_memory_type,
                                                  std::int64_t* actual_memory_type_id) noexcept {
        auto* const state = reinterpret_cast<struct state*>(userp);
        *buffer_userp = state;

        if (!byte_size) {
            *buffer = nullptr;
            return nullptr;
        }

        switch (memory_type) {
            case TRITONSERVER_MEMORY_CPU_PINNED:  // Fallthrough
            case TRITONSERVER_MEMORY_GPU: {
                auto cuda_error = cudaSetDevice(memory_type_id);
                if (cuda_error != cudaSuccess) {
                    std::ostringstream buffer;
                    buffer << service_name
                           << ": Failed in `cudaSetDevice`` while allocating for response: "
                           << cudaGetErrorString(cuda_error);
                    return vtriton::the_shim.ErrorNew(TRITONSERVER_ERROR_UNAVAILABLE,
                                                      buffer.str().c_str());
                }
                cuda_error = (memory_type == TRITONSERVER_MEMORY_CPU_PINNED)
                                 ? cudaHostAlloc(buffer, byte_size, cudaHostAllocPortable)
                                 : cudaMalloc(buffer, byte_size);
                if (cuda_error != cudaSuccess) {
                    std::ostringstream buffer;
                    buffer << service_name
                           << ": Failed in `cuda[HostA|Ma]lloc` while allocating for response: "
                           << cudaGetErrorString(cuda_error);
                    return vtriton::the_shim.ErrorNew(TRITONSERVER_ERROR_UNAVAILABLE,
                                                      buffer.str().c_str());
                }
                break;
            }
            case TRITONSERVER_MEMORY_CPU:  // Fallthrough
            default: {
                memory_type = TRITONSERVER_MEMORY_CPU;
                *buffer = std::malloc(byte_size);
                if (!*buffer) {
                    std::ostringstream buffer;
                    buffer << service_name
                           << ": Failed in `std::malloc` while allocating for response";
                    return vtriton::the_shim.ErrorNew(TRITONSERVER_ERROR_UNAVAILABLE,
                                                      buffer.str().c_str());
                }
                break;
            }
        }

        // We made it!
        *buffer_userp = nullptr;  // userp?
        *actual_memory_type = memory_type;
        *actual_memory_type_id = memory_type_id;

        return nullptr;
    }

    static TRITONSERVER_Error* deallocate_response_(TRITONSERVER_ResponseAllocator* allocator,
                                                    void* buffer,
                                                    void* buffer_userp,
                                                    std::size_t byte_size,
                                                    TRITONSERVER_MemoryType memory_type,
                                                    std::int64_t memory_type_id) noexcept {
        auto* const state = reinterpret_cast<struct state*>(buffer_userp);

        switch (memory_type) {
            case TRITONSERVER_MEMORY_CPU_PINNED:  // Fallthrough
            case TRITONSERVER_MEMORY_GPU: {
                auto cuda_error = cudaSetDevice(memory_type_id);
                if (cuda_error != cudaSuccess) {
                    std::cerr << service_name
                              << ": Failed to obtain cuda device when deallocating response data: `"
                              << cudaGetErrorString(cuda_error) << "` - terminating" << std::endl;
                    std::abort();
                }
                auto cudaFreeFn =
                    (memory_type == TRITONSERVER_MEMORY_CPU_PINNED) ? cudaFreeHost : cudaFree;
                cuda_error = cudaFreeFn(buffer);
                if (cuda_error != cudaSuccess) {
                    std::cerr << service_name
                              << ": Failed cudaFree[host] when deallocating response data: `"
                              << cudaGetErrorString(cuda_error) << "` - terminating" << std::endl;
                    std::abort();
                }
                break;
            }
            case TRITONSERVER_MEMORY_CPU: {
                std::free(buffer);
                break;
            }
            default: {
                std::cerr << service_name
                          << ": Cannot honor request to deallocate unknown MemoryType "
                          << memory_type << " - terminating" << std::endl;
                std::abort();
            }
        }

        return nullptr;
    }

    vtriton::unique_ptr<TRITONSERVER_InferenceRequest> get_inference_request_(
        const std::shared_ptr<struct state>& state) {
        vtriton::unique_ptr<TRITONSERVER_InferenceRequest> result;
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            if (!state->inference_requests.empty()) {
                result = std::move(state->inference_requests.top());
                state->inference_requests.pop();
                return result;
            }
        }

        // TODO: Should model version be a config parameter?
        result = vtriton::make_unique<TRITONSERVER_InferenceRequest>(
            state->server.get(), state->model_name.c_str(), -1);

        vtriton::call(vtriton::the_shim.InferenceRequestSetReleaseCallback)(
            result.get(), &release_inference_request_, state.get());

        return result;
    }

    static void release_inference_request_(TRITONSERVER_InferenceRequest* request,
                                           const uint32_t flags,
                                           void* userp) noexcept {
        if (flags != TRITONSERVER_REQUEST_RELEASE_ALL) {
            // TODO: Log something
            std::abort();
        }

        try {
            auto taken = vtriton::take_unique(request);
            vtriton::call(vtriton::the_shim.InferenceRequestRemoveAllInputs)(taken.get());
            vtriton::call(vtriton::the_shim.InferenceRequestRemoveAllRequestedOutputs)(taken.get());
            auto* const state = reinterpret_cast<struct state*>(userp);
            std::unique_lock<std::mutex> lock(state->mutex);
            state->inference_requests.push(std::move(taken));
        } catch (...) {
            // TODO: log?
        }
    }

    using cuda_unique_ptr = std::unique_ptr<void, decltype(cuda_deleter)>;

    class inference_request_input_visitor_ : public boost::static_visitor<cuda_unique_ptr> {
       public:
        inference_request_input_visitor_(const std::string* name,
                                         TRITONSERVER_InferenceRequest* request,
                                         TRITONSERVER_MemoryType memory_type,
                                         std::int64_t memory_type_id)
            : name_(name),
              request_(request),
              memory_type_(memory_type),
              memory_type_id_(memory_type_id) {}

        template <typename T>
        cuda_unique_ptr operator()(const T& mlmodel_tensor) const {
            std::cout << "XXX ACM Attaching " << name_ << ": [";
            for (size_t i = 0; i != mlmodel_tensor.shape().size(); ++i) {
                std::cout << mlmodel_tensor.shape()[i] << ", ";
            }
            std::cout << "]" << std::endl;
            std::vector<std::size_t> revised_shape(mlmodel_tensor.shape());
            // TODO: We need real metadata!
            if (mlmodel_tensor.shape().size() == 1) {
                revised_shape.assign(4, 0UL);
                revised_shape[0] = 1;
                revised_shape[1] = 640;
                revised_shape[2] = 480;
                revised_shape[3] = 3;

                std::cout << "XXX ACM Revised shape for " << name_ << ": [";
                for (size_t i = 0; i != revised_shape.size(); ++i) {
                    std::cout << revised_shape[i] << ", ";
                }
                std::cout << "]" << std::endl;
            }
            vtriton::call(vtriton::the_shim.InferenceRequestAddInput)(
                request_,
                name_->c_str(),
                triton_datatype_for_(mlmodel_tensor),
                reinterpret_cast<const int64_t*>(&revised_shape[0]),
                revised_shape.size());

            const auto* const mlmodel_data_begin =
                reinterpret_cast<const unsigned char*>(mlmodel_tensor.data());
            const auto* const mlmodel_data_end = reinterpret_cast<const unsigned char*>(
                mlmodel_tensor.data() + mlmodel_tensor.size());
            const auto mlmodel_data_size =
                static_cast<size_t>(mlmodel_data_end - mlmodel_data_begin);

            void* alloc = nullptr;
            const void* data = nullptr;
            cuda_unique_ptr result(nullptr, cuda_deleter);
            switch (memory_type_) {
                case TRITONSERVER_MEMORY_GPU: {
                    auto cuda_error = cudaMalloc(&alloc, mlmodel_data_size);
                    cuda_error = cudaMemcpy(
                        alloc, mlmodel_data_begin, mlmodel_data_size, cudaMemcpyHostToDevice);
                    // TODO: check cuda errors
                    result.reset(alloc);
                    data = alloc;
                    break;
                }
                case TRITONSERVER_MEMORY_CPU_PINNED: {
                    auto cuda_error =
                        cudaHostAlloc(&alloc, mlmodel_data_size, cudaHostAllocPortable);
                    cuda_error = cudaMemcpy(
                        alloc, mlmodel_data_begin, mlmodel_data_size, cudaMemcpyHostToHost);
                    // TODO: check cuda_errors
                    result.reset(alloc);
                    data = alloc;
                    break;
                }
                case TRITONSERVER_MEMORY_CPU:
                default: {
                    data = mlmodel_data_begin;
                    break;
                }
            }

            vtriton::call(vtriton::the_shim.InferenceRequestAppendInputData)(
                request_, name_->c_str(), data, mlmodel_data_size, memory_type_, memory_type_id_);

            return result;
        }

       private:
        template <typename T>
        using tv = vsdk::MLModelService::tensor_view<T>;

        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::int8_t>& t) {
            return TRITONSERVER_TYPE_INT8;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::uint8_t>& t) {
            return TRITONSERVER_TYPE_UINT8;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::int16_t>& t) {
            return TRITONSERVER_TYPE_INT16;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::uint16_t>& t) {
            return TRITONSERVER_TYPE_UINT16;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::int32_t>& t) {
            return TRITONSERVER_TYPE_INT32;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::uint32_t>& t) {
            return TRITONSERVER_TYPE_UINT32;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::int64_t>& t) {
            return TRITONSERVER_TYPE_INT64;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<std::uint64_t>& t) {
            return TRITONSERVER_TYPE_UINT64;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<float>& t) {
            return TRITONSERVER_TYPE_FP32;
        }
        static TRITONSERVER_DataType triton_datatype_for_(const tv<double>& t) {
            return TRITONSERVER_TYPE_FP64;
        }

        const std::string* name_;
        TRITONSERVER_InferenceRequest* request_;
        TRITONSERVER_MemoryType memory_type_;
        std::int64_t memory_type_id_;
    };

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

        // As above, but inverted.
        std::unordered_map<std::string, std::string> input_name_remappings_reversed;
        std::unordered_map<std::string, std::string> output_name_remappings_reversed;

        vtriton::unique_ptr<TRITONSERVER_ResponseAllocator> allocator;
        vtriton::unique_ptr<TRITONSERVER_Server> server;

        std::mutex mutex;
        std::stack<vtriton::unique_ptr<TRITONSERVER_InferenceRequest>> inference_requests;
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
    std::cerr << "ERROR: A std::exception was thrown from `serve`: " << ex.what() << std::endl;
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "ERROR: An unknown exception was thrown from `serve`" << std::endl;
    return EXIT_FAILURE;
}

}  // namespace

extern "C" int example_mlmodelservice_triton_serve(vtriton::shim* shim, const char* sock) {
    vtriton::the_shim = *shim;
    return serve(sock);
}
