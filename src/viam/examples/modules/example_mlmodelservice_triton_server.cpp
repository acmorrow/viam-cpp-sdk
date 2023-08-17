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

#include <cuda_runtime_api.h>

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

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <viam/sdk/components/component.hpp>
#include <viam/sdk/config/resource.hpp>
#include <viam/sdk/module/service.hpp>
#include <viam/sdk/registry/registry.hpp>
#include <viam/sdk/rpc/server.hpp>
#include <viam/sdk/services/mlmodel/mlmodel.hpp>
#include <viam/sdk/services/mlmodel/server.hpp>

#include "vtriton.hpp"

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
        // std::cout << "XXX ACM MLModelServiceTriton: recieved `infer` invocation" << std::endl;
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

        auto error = vtriton::the_shim.InferenceResponseError(inference_response.get());
        if (error) {
            std::ostringstream buffer;
            buffer << ": Triton Server Inference Error: "
                   << vtriton::the_shim.ErrorCodeString(error) << " - "
                   << vtriton::the_shim.ErrorMessage(error);
            std::cerr << "XXX ACM Inference Failed!" << buffer.str() << std::endl;
            throw std::runtime_error(buffer.str());
        }

        // std::cerr << "XXX ACM constructing inference_result_type" << std::endl;
        //  TODO: Comment me.
        struct inference_result_type {
            std::shared_ptr<struct state> state;
            decltype(inference_response) ir;
            std::vector<std::unique_ptr<unsigned char[]>> bufs;
            named_tensor_views ntvs;
        };
        auto inference_result = std::make_shared<inference_result_type>();

        std::uint32_t outputs = 0;
        vtriton::call(vtriton::the_shim.InferenceResponseOutputCount)(inference_response.get(),
                                                                      &outputs);

        // std::cerr << "XXX ACM counted outputs: " << outputs << std::endl;
        for (decltype(outputs) output = 0; output != outputs; ++output) {
            // std::cerr << "XXX ACM working output: " << output << std::endl;

            const char* output_name_cstr;
            TRITONSERVER_DataType data_type;
            const std::int64_t* shape;
            std::uint64_t shape_size;
            const void* data;
            std::size_t data_bytes;
            TRITONSERVER_MemoryType memory_type;
            std::int64_t memory_type_id;
            void* userp;

            vtriton::call(vtriton::the_shim.InferenceResponseOutput)(inference_response.get(),
                                                                     output,
                                                                     &output_name_cstr,
                                                                     &data_type,
                                                                     &shape,
                                                                     &shape_size,
                                                                     &data,
                                                                     &data_bytes,
                                                                     &memory_type,
                                                                     &memory_type_id,
                                                                     &userp);

            if (!output_name_cstr) {
                // TODO: Log this? Return an error?
                continue;
            }
            // std::cerr << "XXX ACM output is named: " << output_name_cstr << std::endl;

            // TODO: Can we avoid some string copies here?
            std::string output_name_string(output_name_cstr);
            const std::string* output_name = &output_name_string;
            const auto where = state->output_name_remappings.find(output_name_string);
            if (where != state->output_name_remappings.end()) {
                output_name = &where->second;
            }
            // std::cerr << "XXX ACM output is renamed: " << *output_name << std::endl;
            //  TODO: Ignore outputs not in metadata?

            // If the memory is on the GPU we need to copy it out, since the higher
            // level doesn't know that it can't just memcpy.
            //
            // TODO: We could save a copy here if we got smarter. maybe.
            if (memory_type == TRITONSERVER_MEMORY_GPU) {
                // std::cerr << "XXX ACM need to copy out of GPU" << std::endl;
                inference_result->bufs.push_back(std::make_unique<unsigned char[]>(data_bytes));
                auto allocated = reinterpret_cast<void*>(inference_result->bufs.back().get());
                const auto cuda_error =
                    cudaMemcpy(allocated, data, data_bytes, cudaMemcpyDeviceToHost);
                // TODO: cehck cuda_error
                data = allocated;
                // std::cerr << "XXX ACM did gpu copy " << cuda_error << std::endl;
            }

            // std::cerr << "XXX ACM making shape vector of size " << shape_size << std::endl;
            std::vector<std::size_t> shape_vector;
            shape_vector.reserve(shape_size);
            for (size_t i = 0; i != shape_size; ++i) {
                auto val = shape[i];
                if (val < 0) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model returned negative value " << val
                           << " in shape for output " << *output_name;
                    throw std::runtime_error(buffer.str());
                }
                shape_vector.push_back(static_cast<std::size_t>(val));
            }

            // std::cerr << "XXX ACM making tensor view " << std::endl;
            auto tv = make_tensor_view_(data_type, data, data_bytes, std::move(shape_vector));

            // TODO: Could conditionally avoid a name copy in the case where
            // we didn't get a name remapping
            // std::cerr << "XXX ACM emplacing result" << std::endl;
            inference_result->ntvs.emplace(*output_name, std::move(tv));
            // std::cerr << "XXX ACM done with output " << output << std::endl;
        }

        // Keep the lease on `state` and move ownership of `inference_response` into
        // `inference_result`. Otherwise, the result would return to the pool and our
        // views would no longer be valid.
        // std::cerr << "XXX ACM stashing state and IR" << std::endl;
        inference_result->state = std::move(state);
        inference_result->ir = std::move(inference_response);

        // Finally, construct an aliasing shared_ptr which appears to
        // the caller as a shared_ptr to views, but in fact manages
        // the lifetime of the inference_result. When the
        // inference_result object is destroyed, we will return
        // the response to the pool.
        auto* const ntvs = &inference_result->ntvs;
        // std::cerr << "XXX ACM returning from infer" << std::endl;
        return {std::move(inference_result), ntvs};

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
        // std::cout << "XXX ACM MLModelServiceTriton: recieved `metadata` invocation" << std::endl;
        //  Just return a copy of our metadata from leased state.
        const auto state = lease_state_();
        return lease_state_()->metadata;
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
                    // std::cout << "XXX ACM Triton Server is live and ready" << std::endl;
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        if (!result) {
            // std::cerr << "XXX ACM Triton Server did not become live and ready" << std::endl;
            throw std::runtime_error("Triton Server did not become live and ready within 30s");
        }

        // TODO: Need to clean up this message object
        TRITONSERVER_Message* model_metadata_message = nullptr;
        vtriton::call(vtriton::the_shim.ServerModelMetadata)(
            server.get(), state->model_name.c_str(), -1, &model_metadata_message);

        // TODO: Model version here from config. And where else?
        const char* model_metadata_json_bytes;
        size_t model_metadata_json_size;
        vtriton::call(vtriton::the_shim.MessageSerializeToJson)(
            model_metadata_message, &model_metadata_json_bytes, &model_metadata_json_size);

        rapidjson::Document model_metadata_json;
        model_metadata_json.Parse(model_metadata_json_bytes, model_metadata_json_size);
        if (model_metadata_json.HasParseError()) {
            std::ostringstream buffer;
            buffer << service_name
                   << ": Failed parsing model metadata returned by triton server at offset "
                   << model_metadata_json.GetErrorOffset() << ": "
                   << rapidjson::GetParseError_En(model_metadata_json.GetParseError());
            throw std::runtime_error(buffer.str());
        }

        const auto populate_tensor_infos = [&model_metadata_json](const auto& array,
                                                                  const auto& name_remappings,
                                                                  auto* tensor_infos) {
            static const std::map<std::string, MLModelService::tensor_info::data_types>
                datatype_map = {{"UINT8", MLModelService::tensor_info::data_types::k_uint8},
                                {"UINT16", MLModelService::tensor_info::data_types::k_uint16},
                                {"UINT32", MLModelService::tensor_info::data_types::k_uint32},
                                {"UINT64", MLModelService::tensor_info::data_types::k_uint64},
                                {"INT8", MLModelService::tensor_info::data_types::k_int8},
                                {"INT16", MLModelService::tensor_info::data_types::k_int16},
                                {"INT32", MLModelService::tensor_info::data_types::k_int32},
                                {"INT64", MLModelService::tensor_info::data_types::k_int64},
                                {"FP32", MLModelService::tensor_info::data_types::k_float32},
                                {"FP64", MLModelService::tensor_info::data_types::k_float64}};

            for (const auto& element : array.GetArray()) {
                if (!element.IsObject()) {
                    std::ostringstream buffer;
                    buffer << service_name
                           << ": Model metadata array is expected to contain object fields";
                    throw std::runtime_error(buffer.str());
                }
                if (!element.HasMember("name")) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model metadata entry has no `name` field";
                    throw std::runtime_error(buffer.str());
                }
                const auto& name_element = element["name"];
                if (!name_element.IsString()) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model metadata entry `name` field is not a string";
                    throw std::runtime_error(buffer.str());
                }
                const auto name = name_element.GetString();
                auto viam_name = name;
                const auto name_remappings_where = name_remappings.find(name);
                if (name_remappings_where != name_remappings.end()) {
                    viam_name = name_remappings_where->second.c_str();
                }

                if (!element.HasMember("datatype")) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model metadata entry for tensor `" << name
                           << "` does not have a `datatype` field";
                    throw std::runtime_error(buffer.str());
                }
                const auto& datatype_element = element["datatype"];
                if (!datatype_element.IsString()) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model metadata entry `datatype` field for tensor `"
                           << name << "` is not a string";
                    throw std::runtime_error(buffer.str());
                }
                const auto& triton_datatype = datatype_element.GetString();
                const auto datatype_map_where = datatype_map.find(triton_datatype);
                if (datatype_map_where == datatype_map.end()) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model metadata entry `datatype` field for tensor `"
                           << name << "` contains unsupported data type `" << triton_datatype
                           << "`";
                    throw std::runtime_error(buffer.str());
                }
                const auto viam_datatype = datatype_map_where->second;

                if (!element.HasMember("shape")) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model metadata entry for tensor `" << name
                           << "` does not have a `shape` field";
                    throw std::runtime_error(buffer.str());
                }
                const auto& shape_element = element["shape"];
                if (!shape_element.IsArray()) {
                    std::ostringstream buffer;
                    buffer << service_name << ": Model metadata entry `shape` field for tensor `"
                           << name << "` is not an array";
                    throw std::runtime_error(buffer.str());
                }

                std::vector<int> shape;
                for (const auto& shape_element_entry : shape_element.GetArray()) {
                    if (!shape_element_entry.IsInt()) {
                        std::ostringstream buffer;
                        buffer << service_name
                               << ": Model metadata entry `shape` field for tensor `" << name
                               << "` contained a non-integer value";
                    }
                    shape.push_back(shape_element_entry.GetInt());
                }

                std::cout << "XXX ACM ADDING METADATA" <<
                    "triton name: " << name << " "
                    "viam name: " << viam_name << " "
                    "triton datatype: " << triton_datatype << " "
                    "viam datatype: " << (int)viam_datatype << " "
                    "shape: [";
                for (const auto& elt : shape) {
                    std::cout << elt << ", ";
                }
                std::cout << "]" << std::endl;

                tensor_infos->push_back({
                    // `name`
                    viam_name,

                    // `description`
                    "",

                    // `data_type`
                    viam_datatype,

                    // `shape`
                    std::move(shape),
                });
            }
        };

        if (!model_metadata_json.HasMember("inputs")) {
            std::ostringstream buffer;
            buffer << service_name << ": Model metadata does not include an `inputs` field";
            throw std::runtime_error(buffer.str());
        }
        const auto& inputs = model_metadata_json["inputs"];
        if (!inputs.IsArray()) {
            std::ostringstream buffer;
            buffer << service_name << ": Model metadata `inputs` field is not an array";
            throw std::runtime_error(buffer.str());
        }
        populate_tensor_infos(inputs, state->input_name_remappings, &state->metadata.inputs);

        if (!model_metadata_json.HasMember("outputs")) {
            std::ostringstream buffer;
            buffer << service_name << ": Model metadata does not include an `outputs` field";
            throw std::runtime_error(buffer.str());
        }
        const auto& outputs = model_metadata_json["outputs"];
        if (!outputs.IsArray()) {
            std::ostringstream buffer;
            buffer << service_name << ": Model metadata `outputs` field is not an array";
            throw std::runtime_error(buffer.str());
        }
        populate_tensor_infos(outputs, state->output_name_remappings, &state->metadata.outputs);

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
            std::vector<std::size_t> revised_shape(mlmodel_tensor.shape());
            // TODO: We need real metadata!
            if (mlmodel_tensor.shape().size() == 1) {
                revised_shape.assign(4, 0UL);
                revised_shape[0] = 1;
                revised_shape[1] = 640;
                revised_shape[2] = 480;
                revised_shape[3] = 3;
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

    MLModelService::tensor_views make_tensor_view_(TRITONSERVER_DataType data_type,
                                                   const void* data,
                                                   size_t data_bytes,
                                                   std::vector<std::size_t>&& shape_vector) {
        switch (data_type) {
            case TRITONSERVER_TYPE_INT8: {
                // std::cerr << "XXX ACM Making std::int8_t tensor" << std::endl;
                return make_tensor_view_t_<std::int8_t>(data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_UINT8: {
                // std::cerr << "XXX ACM Making std::uint8_t tensor" << std::endl;
                return make_tensor_view_t_<std::uint8_t>(data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_INT16: {
                // std::cerr << "XXX ACM Making std::int16_t tensor" << std::endl;
                return make_tensor_view_t_<std::int16_t>(data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_UINT16: {
                // std::cerr << "XXX ACM Making std::uint16_t tensor" << std::endl;
                return make_tensor_view_t_<std::uint16_t>(
                    data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_INT32: {
                // std::cerr << "XXX ACM Making std::int32_t tensor" << std::endl;
                return make_tensor_view_t_<std::int32_t>(data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_UINT32: {
                // std::cerr << "XXX ACM Making std::uint32_t tensor" << std::endl;
                return make_tensor_view_t_<std::uint32_t>(
                    data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_INT64: {
                // std::cerr << "XXX ACM Making std::int64_t tensor" << std::endl;
                return make_tensor_view_t_<std::int64_t>(data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_UINT64: {
                // std::cerr << "XXX ACM Making std::uint64_t tensor" << std::endl;
                return make_tensor_view_t_<std::uint64_t>(
                    data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_FP32: {
                // std::cerr << "XXX ACM Making float tensor" << std::endl;
                return make_tensor_view_t_<float>(data, data_bytes, std::move(shape_vector));
            }
            case TRITONSERVER_TYPE_FP64: {
                // std::cerr << "XXX ACM Making double tensor" << std::endl;
                return make_tensor_view_t_<double>(data, data_bytes, std::move(shape_vector));
            }
            default: {
                std::ostringstream buffer;
                buffer << service_name
                       << ": Model returned unsupported tflite data type: " << data_type;
                throw std::invalid_argument(buffer.str());
            }
        }
    }

    template <typename T>
    MLModelService::tensor_views make_tensor_view_t_(const void* data,
                                                     size_t data_bytes,
                                                     std::vector<std::size_t>&& shape_vector) {
        const auto* const typed_data = reinterpret_cast<const T*>(data);
        const auto typed_size = data_bytes / sizeof(*typed_data);
        // std::cerr << "XXX ACM Data " << data << " of " << data_bytes << " bytes is " <<
        // typed_size << " elements" << std::endl; std::cerr << "XXX ACM shape vector is: ["; for
        // (const auto elt : shape_vector) {
        //     std::cerr << elt << ", ";
        // }
        // std::cerr << "]" << std::endl;
        // std::cerr << "XXX ACM first 10 elements of data vector is: [";
        // for (size_t i = 0; i < 10 && i < typed_size; ++i) {
        //     std::cerr << typed_data[i] << ", ";
        // }
        // std::cerr << "]" << std::endl;

        return MLModelService::make_tensor_view(typed_data, typed_size, std::move(shape_vector));
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
            // std::cout << "XXX ACM Destroying State" << std::endl;
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

        // Metadata about input and output tensors that was extracted
        // during configuration. Callers need this in order to know
        // how to interact with the service.
        struct MLModelService::metadata metadata;

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

        // The response allocator and server this state will use.
        //
        // TODO: Pooling?
        vtriton::unique_ptr<TRITONSERVER_ResponseAllocator> allocator;
        vtriton::unique_ptr<TRITONSERVER_Server> server;

        // Inference requests are pooled and reused.
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
