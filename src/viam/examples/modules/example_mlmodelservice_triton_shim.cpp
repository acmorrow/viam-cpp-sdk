#include "vtriton.hpp"
#include <iostream>
void example_mlmodelservice_triton_shim_init(vtriton::shim* s) {
    s->ApiVersion = &TRITONSERVER_ApiVersion;
    std::cout << "XXX ACM INSIDE shim: " << ((void*)(s->ApiVersion)) << std::endl;
    std::cout << "XXX ACM INSIDE pshim: " << ((void**)&(s->ApiVersion)) << std::endl;
    std::cout << "XXX ACM INSIDE real:" << ((void*)TRITONSERVER_ApiVersion) << std::endl;

    s->ErrorCodeString = &TRITONSERVER_ErrorCodeString;
    s->ErrorDelete = &TRITONSERVER_ErrorDelete;
    s->ErrorMessage = &TRITONSERVER_ErrorMessage;
    s->ErrorNew = &TRITONSERVER_ErrorNew;
    s->ServerOptionsDelete = &TRITONSERVER_ServerOptionsDelete;
    s->ServerOptionsNew = &TRITONSERVER_ServerOptionsNew;
    s->ServerDelete = &TRITONSERVER_ServerDelete;
    s->ServerNew = &TRITONSERVER_ServerNew;
    s->ServerOptionsSetBackendDirectory = &TRITONSERVER_ServerOptionsSetBackendDirectory;
    s->ServerOptionsSetLogVerbose = &TRITONSERVER_ServerOptionsSetLogVerbose;
    s->ServerOptionsSetMinSupportedComputeCapability = &TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability;
    s->ServerOptionsSetModelRepositoryPath = &TRITONSERVER_ServerOptionsSetModelRepositoryPath;
    s->ServerOptionsSetStrictModelConfig = &TRITONSERVER_ServerOptionsSetStrictModelConfig;
}
