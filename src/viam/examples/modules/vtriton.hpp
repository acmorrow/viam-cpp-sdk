#pragma once

#include <stdexcept>
#include <memory>
#include <sstream>

#include <triton/core/tritonserver.h>

namespace vtriton {

struct shim {
    decltype(TRITONSERVER_ApiVersion)* ApiVersion = nullptr;
    decltype(TRITONSERVER_ErrorCodeString)* ErrorCodeString = nullptr;
    decltype(TRITONSERVER_ErrorDelete)* ErrorDelete = nullptr;
    decltype(TRITONSERVER_ErrorMessage)* ErrorMessage = nullptr;
    decltype(TRITONSERVER_ErrorNew)* ErrorNew = nullptr;
    decltype(TRITONSERVER_ServerOptionsDelete)* ServerOptionsDelete = nullptr;
    decltype(TRITONSERVER_ServerOptionsNew)* ServerOptionsNew = nullptr;
    decltype(TRITONSERVER_ServerDelete)* ServerDelete = nullptr;
    decltype(TRITONSERVER_ServerNew)* ServerNew = nullptr;
    decltype(TRITONSERVER_ServerOptionsSetBackendDirectory)* ServerOptionsSetBackendDirectory = nullptr;
    decltype(TRITONSERVER_ServerOptionsSetLogVerbose)* ServerOptionsSetLogVerbose = nullptr;
    decltype(TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability)*
    ServerOptionsSetMinSupportedComputeCapability = nullptr;
    decltype(TRITONSERVER_ServerOptionsSetModelRepositoryPath)* ServerOptionsSetModelRepositoryPath =
        nullptr;
    decltype(TRITONSERVER_ServerOptionsSetStrictModelConfig)* ServerOptionsSetStrictModelConfig =
        nullptr;
};

shim the_shim;

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
    template <class... Args>
    static auto ctor(Args&&... args) {
        return the_shim.ErrorNew(std::forward<Args>(args)...);
    }
    template <class... Args>
    static auto dtor(Args&&... args) {
        return the_shim.ErrorDelete(std::forward<Args>(args)...);
    }
};

template <>
struct lifecycle_traits<TRITONSERVER_ServerOptions> {
    using value_type = TRITONSERVER_ServerOptions;
    template <class... Args>
    static auto ctor(Args&&... args) {
        TRITONSERVER_ServerOptions* opts = nullptr;
        call(the_shim.ServerOptionsNew)(&opts, std::forward<Args>(args)...);
        return opts;
    };
    template <class... Args>
    static auto dtor(Args&&... args) {
        call(the_shim.ServerOptionsDelete)(std::forward<Args>(args)...);
    };
};

template <>
struct lifecycle_traits<TRITONSERVER_Server> {
    using value_type = TRITONSERVER_Server;
    template <class... Args>
    static auto ctor(Args&&... args) {
        TRITONSERVER_Server* server = nullptr;
        call(the_shim.ServerNew)(&server, std::forward<Args>(args)...);
        return server;
    };
    template <class... Args>
    static auto dtor(Args&&... args) {
        call(the_shim.ServerDelete)(std::forward<Args>(args)...);
    };
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
            buffer << ": Triton Server Error: " << the_shim.ErrorCodeString(error.get()) << " - "
                   << the_shim.ErrorMessage(error.get());
            throw Stdex(buffer.str());
        }
    };
}

}  // namespace vtriton

extern "C" void example_mlmodelservice_triton_shim_init(vtriton::shim* s);
