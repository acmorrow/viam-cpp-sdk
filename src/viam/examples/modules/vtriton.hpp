#pragma once

#include <dlfcn.h>

#include <triton/core/tritonserver.h>

namespace vtriton {

namespace shim {
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

inline bool setup(const char* library) {
    auto handle = dlmopen(LM_ID_NEWLM, library, RTLD_NOW);
    //auto handle = dlopen(library, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        std::cout << "XXXXXXXXXXX ACM OOPS dlmopen: " << dlerror() << std::endl;
        return false;
    }

    vtriton::shim::ApiVersion = reinterpret_cast<decltype(vtriton::shim::ApiVersion)>(
        dlsym(handle, "TRITONSERVER_ApiVersion"));
    if (!vtriton::shim::ApiVersion) {
        return false;
    }

    vtriton::shim::ErrorCodeString = reinterpret_cast<decltype(vtriton::shim::ErrorCodeString)>(
        dlsym(handle, "TRITONSERVER_ErrorCodeString"));
    if (!vtriton::shim::ErrorCodeString) {
        return false;
    }

    vtriton::shim::ErrorDelete = reinterpret_cast<decltype(vtriton::shim::ErrorDelete)>(
        dlsym(handle, "TRITONSERVER_ErrorDelete"));
    if (!vtriton::shim::ErrorDelete) {
        return false;
    }

    vtriton::shim::ErrorMessage = reinterpret_cast<decltype(vtriton::shim::ErrorMessage)>(
        dlsym(handle, "TRITONSERVER_ErrorMessage"));
    if (!vtriton::shim::ErrorMessage) {
        return false;
    }

    vtriton::shim::ErrorNew =
        reinterpret_cast<decltype(vtriton::shim::ErrorNew)>(dlsym(handle, "TRITONSERVER_ErrorNew"));
    if (!vtriton::shim::ErrorNew) {
        return false;
    }

    vtriton::shim::ServerOptionsDelete =
        reinterpret_cast<decltype(vtriton::shim::ServerOptionsDelete)>(
            dlsym(handle, "TRITONSERVER_ServerOptionsDelete"));
    if (!vtriton::shim::ServerOptionsDelete) {
        return false;
    }

    vtriton::shim::ServerOptionsNew = reinterpret_cast<decltype(vtriton::shim::ServerOptionsNew)>(
        dlsym(handle, "TRITONSERVER_ServerOptionsNew"));
    if (!vtriton::shim::ServerOptionsNew) {
        return false;
    }

    vtriton::shim::ServerDelete = reinterpret_cast<decltype(vtriton::shim::ServerDelete)>(
        dlsym(handle, "TRITONSERVER_ServerDelete"));
    if (!vtriton::shim::ServerDelete) {
        return false;
    }

    vtriton::shim::ServerNew = reinterpret_cast<decltype(vtriton::shim::ServerNew)>(
        dlsym(handle, "TRITONSERVER_ServerNew"));
    if (!vtriton::shim::ServerNew) {
        return false;
    }

    vtriton::shim::ServerOptionsSetBackendDirectory =
        reinterpret_cast<decltype(vtriton::shim::ServerOptionsSetBackendDirectory)>(
            dlsym(handle, "TRITONSERVER_ServerOptionsSetBackendDirectory"));
    if (!vtriton::shim::ServerOptionsSetBackendDirectory) {
        return false;
    }

    vtriton::shim::ServerOptionsSetLogVerbose =
        reinterpret_cast<decltype(vtriton::shim::ServerOptionsSetLogVerbose)>(
            dlsym(handle, "TRITONSERVER_ServerOptionsSetLogVerbose"));
    if (!vtriton::shim::ServerOptionsSetLogVerbose) {
        return false;
    }

    vtriton::shim::ServerOptionsSetMinSupportedComputeCapability =
        reinterpret_cast<decltype(vtriton::shim::ServerOptionsSetMinSupportedComputeCapability)>(
            dlsym(handle, "TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability"));
    if (!vtriton::shim::ServerOptionsSetMinSupportedComputeCapability) {
        return false;
    }

    vtriton::shim::ServerOptionsSetModelRepositoryPath =
        reinterpret_cast<decltype(vtriton::shim::ServerOptionsSetModelRepositoryPath)>(
            dlsym(handle, "TRITONSERVER_ServerOptionsSetModelRepositoryPath"));
    if (!vtriton::shim::ServerOptionsSetModelRepositoryPath) {
        return false;
    }

    vtriton::shim::ServerOptionsSetStrictModelConfig =
        reinterpret_cast<decltype(vtriton::shim::ServerOptionsSetStrictModelConfig)>(
            dlsym(handle, "TRITONSERVER_ServerOptionsSetStrictModelConfig"));
    if (!vtriton::shim::ServerOptionsSetStrictModelConfig) {
        return false;
    }

    return true;
}
}  // namespace shim

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
        return shim::ErrorNew(std::forward<Args>(args)...);
    }
    template <class... Args>
    static auto dtor(Args&&... args) {
        return shim::ErrorDelete(std::forward<Args>(args)...);
    }
};

template <>
struct lifecycle_traits<TRITONSERVER_ServerOptions> {
    using value_type = TRITONSERVER_ServerOptions;
    template <class... Args>
    static auto ctor(Args&&... args) {
        TRITONSERVER_ServerOptions* opts = nullptr;
        call(shim::ServerOptionsNew)(&opts, std::forward<Args>(args)...);
        return opts;
    };
    template <class... Args>
    static auto dtor(Args&&... args) {
        call(shim::ServerOptionsDelete)(std::forward<Args>(args)...);
    };
};

template <>
struct lifecycle_traits<TRITONSERVER_Server> {
    using value_type = TRITONSERVER_Server;
    template <class... Args>
    static auto ctor(Args&&... args) {
        TRITONSERVER_Server* server = nullptr;
        call(shim::ServerNew)(&server, std::forward<Args>(args)...);
        return server;
    };
    template <class... Args>
    static auto dtor(Args&&... args) {
        call(shim::ServerDelete)(std::forward<Args>(args)...);
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
            buffer << ": Triton Server Error: " << shim::ErrorCodeString(error.get()) << " - "
                   << shim::ErrorMessage(error.get());
            throw Stdex(buffer.str());
        }
    };
}

}  // namespace vtriton
