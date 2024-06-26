// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: component/testecho/v1/testecho.proto

#include "component/testecho/v1/testecho.pb.h"
#include "component/testecho/v1/testecho.grpc.pb.h"

#include <functional>
#include <grpcpp/support/async_stream.h>
#include <grpcpp/support/async_unary_call.h>
#include <grpcpp/impl/channel_interface.h>
#include <grpcpp/impl/client_unary_call.h>
#include <grpcpp/support/client_callback.h>
#include <grpcpp/support/message_allocator.h>
#include <grpcpp/support/method_handler.h>
#include <grpcpp/impl/rpc_service_method.h>
#include <grpcpp/support/server_callback.h>
#include <grpcpp/impl/server_callback_handlers.h>
#include <grpcpp/server_context.h>
#include <grpcpp/impl/service_type.h>
#include <grpcpp/support/sync_stream.h>
namespace viam {
namespace component {
namespace testecho {
namespace v1 {

static const char* TestEchoService_method_names[] = {
  "/viam.component.testecho.v1.TestEchoService/Echo",
  "/viam.component.testecho.v1.TestEchoService/EchoMultiple",
  "/viam.component.testecho.v1.TestEchoService/EchoBiDi",
  "/viam.component.testecho.v1.TestEchoService/Stop",
};

std::unique_ptr< TestEchoService::Stub> TestEchoService::NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options) {
  (void)options;
  std::unique_ptr< TestEchoService::Stub> stub(new TestEchoService::Stub(channel, options));
  return stub;
}

TestEchoService::Stub::Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options)
  : channel_(channel), rpcmethod_Echo_(TestEchoService_method_names[0], options.suffix_for_stats(),::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_EchoMultiple_(TestEchoService_method_names[1], options.suffix_for_stats(),::grpc::internal::RpcMethod::SERVER_STREAMING, channel)
  , rpcmethod_EchoBiDi_(TestEchoService_method_names[2], options.suffix_for_stats(),::grpc::internal::RpcMethod::BIDI_STREAMING, channel)
  , rpcmethod_Stop_(TestEchoService_method_names[3], options.suffix_for_stats(),::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  {}

::grpc::Status TestEchoService::Stub::Echo(::grpc::ClientContext* context, const ::viam::component::testecho::v1::EchoRequest& request, ::viam::component::testecho::v1::EchoResponse* response) {
  return ::grpc::internal::BlockingUnaryCall< ::viam::component::testecho::v1::EchoRequest, ::viam::component::testecho::v1::EchoResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), rpcmethod_Echo_, context, request, response);
}

void TestEchoService::Stub::async::Echo(::grpc::ClientContext* context, const ::viam::component::testecho::v1::EchoRequest* request, ::viam::component::testecho::v1::EchoResponse* response, std::function<void(::grpc::Status)> f) {
  ::grpc::internal::CallbackUnaryCall< ::viam::component::testecho::v1::EchoRequest, ::viam::component::testecho::v1::EchoResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_Echo_, context, request, response, std::move(f));
}

void TestEchoService::Stub::async::Echo(::grpc::ClientContext* context, const ::viam::component::testecho::v1::EchoRequest* request, ::viam::component::testecho::v1::EchoResponse* response, ::grpc::ClientUnaryReactor* reactor) {
  ::grpc::internal::ClientCallbackUnaryFactory::Create< ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_Echo_, context, request, response, reactor);
}

::grpc::ClientAsyncResponseReader< ::viam::component::testecho::v1::EchoResponse>* TestEchoService::Stub::PrepareAsyncEchoRaw(::grpc::ClientContext* context, const ::viam::component::testecho::v1::EchoRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderHelper::Create< ::viam::component::testecho::v1::EchoResponse, ::viam::component::testecho::v1::EchoRequest, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), cq, rpcmethod_Echo_, context, request);
}

::grpc::ClientAsyncResponseReader< ::viam::component::testecho::v1::EchoResponse>* TestEchoService::Stub::AsyncEchoRaw(::grpc::ClientContext* context, const ::viam::component::testecho::v1::EchoRequest& request, ::grpc::CompletionQueue* cq) {
  auto* result =
    this->PrepareAsyncEchoRaw(context, request, cq);
  result->StartCall();
  return result;
}

::grpc::ClientReader< ::viam::component::testecho::v1::EchoMultipleResponse>* TestEchoService::Stub::EchoMultipleRaw(::grpc::ClientContext* context, const ::viam::component::testecho::v1::EchoMultipleRequest& request) {
  return ::grpc::internal::ClientReaderFactory< ::viam::component::testecho::v1::EchoMultipleResponse>::Create(channel_.get(), rpcmethod_EchoMultiple_, context, request);
}

void TestEchoService::Stub::async::EchoMultiple(::grpc::ClientContext* context, const ::viam::component::testecho::v1::EchoMultipleRequest* request, ::grpc::ClientReadReactor< ::viam::component::testecho::v1::EchoMultipleResponse>* reactor) {
  ::grpc::internal::ClientCallbackReaderFactory< ::viam::component::testecho::v1::EchoMultipleResponse>::Create(stub_->channel_.get(), stub_->rpcmethod_EchoMultiple_, context, request, reactor);
}

::grpc::ClientAsyncReader< ::viam::component::testecho::v1::EchoMultipleResponse>* TestEchoService::Stub::AsyncEchoMultipleRaw(::grpc::ClientContext* context, const ::viam::component::testecho::v1::EchoMultipleRequest& request, ::grpc::CompletionQueue* cq, void* tag) {
  return ::grpc::internal::ClientAsyncReaderFactory< ::viam::component::testecho::v1::EchoMultipleResponse>::Create(channel_.get(), cq, rpcmethod_EchoMultiple_, context, request, true, tag);
}

::grpc::ClientAsyncReader< ::viam::component::testecho::v1::EchoMultipleResponse>* TestEchoService::Stub::PrepareAsyncEchoMultipleRaw(::grpc::ClientContext* context, const ::viam::component::testecho::v1::EchoMultipleRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncReaderFactory< ::viam::component::testecho::v1::EchoMultipleResponse>::Create(channel_.get(), cq, rpcmethod_EchoMultiple_, context, request, false, nullptr);
}

::grpc::ClientReaderWriter< ::viam::component::testecho::v1::EchoBiDiRequest, ::viam::component::testecho::v1::EchoBiDiResponse>* TestEchoService::Stub::EchoBiDiRaw(::grpc::ClientContext* context) {
  return ::grpc::internal::ClientReaderWriterFactory< ::viam::component::testecho::v1::EchoBiDiRequest, ::viam::component::testecho::v1::EchoBiDiResponse>::Create(channel_.get(), rpcmethod_EchoBiDi_, context);
}

void TestEchoService::Stub::async::EchoBiDi(::grpc::ClientContext* context, ::grpc::ClientBidiReactor< ::viam::component::testecho::v1::EchoBiDiRequest,::viam::component::testecho::v1::EchoBiDiResponse>* reactor) {
  ::grpc::internal::ClientCallbackReaderWriterFactory< ::viam::component::testecho::v1::EchoBiDiRequest,::viam::component::testecho::v1::EchoBiDiResponse>::Create(stub_->channel_.get(), stub_->rpcmethod_EchoBiDi_, context, reactor);
}

::grpc::ClientAsyncReaderWriter< ::viam::component::testecho::v1::EchoBiDiRequest, ::viam::component::testecho::v1::EchoBiDiResponse>* TestEchoService::Stub::AsyncEchoBiDiRaw(::grpc::ClientContext* context, ::grpc::CompletionQueue* cq, void* tag) {
  return ::grpc::internal::ClientAsyncReaderWriterFactory< ::viam::component::testecho::v1::EchoBiDiRequest, ::viam::component::testecho::v1::EchoBiDiResponse>::Create(channel_.get(), cq, rpcmethod_EchoBiDi_, context, true, tag);
}

::grpc::ClientAsyncReaderWriter< ::viam::component::testecho::v1::EchoBiDiRequest, ::viam::component::testecho::v1::EchoBiDiResponse>* TestEchoService::Stub::PrepareAsyncEchoBiDiRaw(::grpc::ClientContext* context, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncReaderWriterFactory< ::viam::component::testecho::v1::EchoBiDiRequest, ::viam::component::testecho::v1::EchoBiDiResponse>::Create(channel_.get(), cq, rpcmethod_EchoBiDi_, context, false, nullptr);
}

::grpc::Status TestEchoService::Stub::Stop(::grpc::ClientContext* context, const ::viam::component::testecho::v1::StopRequest& request, ::viam::component::testecho::v1::StopResponse* response) {
  return ::grpc::internal::BlockingUnaryCall< ::viam::component::testecho::v1::StopRequest, ::viam::component::testecho::v1::StopResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), rpcmethod_Stop_, context, request, response);
}

void TestEchoService::Stub::async::Stop(::grpc::ClientContext* context, const ::viam::component::testecho::v1::StopRequest* request, ::viam::component::testecho::v1::StopResponse* response, std::function<void(::grpc::Status)> f) {
  ::grpc::internal::CallbackUnaryCall< ::viam::component::testecho::v1::StopRequest, ::viam::component::testecho::v1::StopResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_Stop_, context, request, response, std::move(f));
}

void TestEchoService::Stub::async::Stop(::grpc::ClientContext* context, const ::viam::component::testecho::v1::StopRequest* request, ::viam::component::testecho::v1::StopResponse* response, ::grpc::ClientUnaryReactor* reactor) {
  ::grpc::internal::ClientCallbackUnaryFactory::Create< ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_Stop_, context, request, response, reactor);
}

::grpc::ClientAsyncResponseReader< ::viam::component::testecho::v1::StopResponse>* TestEchoService::Stub::PrepareAsyncStopRaw(::grpc::ClientContext* context, const ::viam::component::testecho::v1::StopRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderHelper::Create< ::viam::component::testecho::v1::StopResponse, ::viam::component::testecho::v1::StopRequest, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), cq, rpcmethod_Stop_, context, request);
}

::grpc::ClientAsyncResponseReader< ::viam::component::testecho::v1::StopResponse>* TestEchoService::Stub::AsyncStopRaw(::grpc::ClientContext* context, const ::viam::component::testecho::v1::StopRequest& request, ::grpc::CompletionQueue* cq) {
  auto* result =
    this->PrepareAsyncStopRaw(context, request, cq);
  result->StartCall();
  return result;
}

TestEchoService::Service::Service() {
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      TestEchoService_method_names[0],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< TestEchoService::Service, ::viam::component::testecho::v1::EchoRequest, ::viam::component::testecho::v1::EchoResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(
          [](TestEchoService::Service* service,
             ::grpc::ServerContext* ctx,
             const ::viam::component::testecho::v1::EchoRequest* req,
             ::viam::component::testecho::v1::EchoResponse* resp) {
               return service->Echo(ctx, req, resp);
             }, this)));
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      TestEchoService_method_names[1],
      ::grpc::internal::RpcMethod::SERVER_STREAMING,
      new ::grpc::internal::ServerStreamingHandler< TestEchoService::Service, ::viam::component::testecho::v1::EchoMultipleRequest, ::viam::component::testecho::v1::EchoMultipleResponse>(
          [](TestEchoService::Service* service,
             ::grpc::ServerContext* ctx,
             const ::viam::component::testecho::v1::EchoMultipleRequest* req,
             ::grpc::ServerWriter<::viam::component::testecho::v1::EchoMultipleResponse>* writer) {
               return service->EchoMultiple(ctx, req, writer);
             }, this)));
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      TestEchoService_method_names[2],
      ::grpc::internal::RpcMethod::BIDI_STREAMING,
      new ::grpc::internal::BidiStreamingHandler< TestEchoService::Service, ::viam::component::testecho::v1::EchoBiDiRequest, ::viam::component::testecho::v1::EchoBiDiResponse>(
          [](TestEchoService::Service* service,
             ::grpc::ServerContext* ctx,
             ::grpc::ServerReaderWriter<::viam::component::testecho::v1::EchoBiDiResponse,
             ::viam::component::testecho::v1::EchoBiDiRequest>* stream) {
               return service->EchoBiDi(ctx, stream);
             }, this)));
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      TestEchoService_method_names[3],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< TestEchoService::Service, ::viam::component::testecho::v1::StopRequest, ::viam::component::testecho::v1::StopResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(
          [](TestEchoService::Service* service,
             ::grpc::ServerContext* ctx,
             const ::viam::component::testecho::v1::StopRequest* req,
             ::viam::component::testecho::v1::StopResponse* resp) {
               return service->Stop(ctx, req, resp);
             }, this)));
}

TestEchoService::Service::~Service() {
}

::grpc::Status TestEchoService::Service::Echo(::grpc::ServerContext* context, const ::viam::component::testecho::v1::EchoRequest* request, ::viam::component::testecho::v1::EchoResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status TestEchoService::Service::EchoMultiple(::grpc::ServerContext* context, const ::viam::component::testecho::v1::EchoMultipleRequest* request, ::grpc::ServerWriter< ::viam::component::testecho::v1::EchoMultipleResponse>* writer) {
  (void) context;
  (void) request;
  (void) writer;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status TestEchoService::Service::EchoBiDi(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::viam::component::testecho::v1::EchoBiDiResponse, ::viam::component::testecho::v1::EchoBiDiRequest>* stream) {
  (void) context;
  (void) stream;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status TestEchoService::Service::Stop(::grpc::ServerContext* context, const ::viam::component::testecho::v1::StopRequest* request, ::viam::component::testecho::v1::StopResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}


}  // namespace viam
}  // namespace component
}  // namespace testecho
}  // namespace v1

