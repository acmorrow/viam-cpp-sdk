// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: service/sensors/v1/sensors.proto

#include "service/sensors/v1/sensors.pb.h"
#include "service/sensors/v1/sensors.grpc.pb.h"

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
namespace service {
namespace sensors {
namespace v1 {

static const char* SensorsService_method_names[] = {
  "/viam.service.sensors.v1.SensorsService/GetSensors",
  "/viam.service.sensors.v1.SensorsService/GetReadings",
  "/viam.service.sensors.v1.SensorsService/DoCommand",
};

std::unique_ptr< SensorsService::Stub> SensorsService::NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options) {
  (void)options;
  std::unique_ptr< SensorsService::Stub> stub(new SensorsService::Stub(channel, options));
  return stub;
}

SensorsService::Stub::Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options)
  : channel_(channel), rpcmethod_GetSensors_(SensorsService_method_names[0], options.suffix_for_stats(),::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_GetReadings_(SensorsService_method_names[1], options.suffix_for_stats(),::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_DoCommand_(SensorsService_method_names[2], options.suffix_for_stats(),::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  {}

::grpc::Status SensorsService::Stub::GetSensors(::grpc::ClientContext* context, const ::viam::service::sensors::v1::GetSensorsRequest& request, ::viam::service::sensors::v1::GetSensorsResponse* response) {
  return ::grpc::internal::BlockingUnaryCall< ::viam::service::sensors::v1::GetSensorsRequest, ::viam::service::sensors::v1::GetSensorsResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), rpcmethod_GetSensors_, context, request, response);
}

void SensorsService::Stub::async::GetSensors(::grpc::ClientContext* context, const ::viam::service::sensors::v1::GetSensorsRequest* request, ::viam::service::sensors::v1::GetSensorsResponse* response, std::function<void(::grpc::Status)> f) {
  ::grpc::internal::CallbackUnaryCall< ::viam::service::sensors::v1::GetSensorsRequest, ::viam::service::sensors::v1::GetSensorsResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_GetSensors_, context, request, response, std::move(f));
}

void SensorsService::Stub::async::GetSensors(::grpc::ClientContext* context, const ::viam::service::sensors::v1::GetSensorsRequest* request, ::viam::service::sensors::v1::GetSensorsResponse* response, ::grpc::ClientUnaryReactor* reactor) {
  ::grpc::internal::ClientCallbackUnaryFactory::Create< ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_GetSensors_, context, request, response, reactor);
}

::grpc::ClientAsyncResponseReader< ::viam::service::sensors::v1::GetSensorsResponse>* SensorsService::Stub::PrepareAsyncGetSensorsRaw(::grpc::ClientContext* context, const ::viam::service::sensors::v1::GetSensorsRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderHelper::Create< ::viam::service::sensors::v1::GetSensorsResponse, ::viam::service::sensors::v1::GetSensorsRequest, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), cq, rpcmethod_GetSensors_, context, request);
}

::grpc::ClientAsyncResponseReader< ::viam::service::sensors::v1::GetSensorsResponse>* SensorsService::Stub::AsyncGetSensorsRaw(::grpc::ClientContext* context, const ::viam::service::sensors::v1::GetSensorsRequest& request, ::grpc::CompletionQueue* cq) {
  auto* result =
    this->PrepareAsyncGetSensorsRaw(context, request, cq);
  result->StartCall();
  return result;
}

::grpc::Status SensorsService::Stub::GetReadings(::grpc::ClientContext* context, const ::viam::service::sensors::v1::GetReadingsRequest& request, ::viam::service::sensors::v1::GetReadingsResponse* response) {
  return ::grpc::internal::BlockingUnaryCall< ::viam::service::sensors::v1::GetReadingsRequest, ::viam::service::sensors::v1::GetReadingsResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), rpcmethod_GetReadings_, context, request, response);
}

void SensorsService::Stub::async::GetReadings(::grpc::ClientContext* context, const ::viam::service::sensors::v1::GetReadingsRequest* request, ::viam::service::sensors::v1::GetReadingsResponse* response, std::function<void(::grpc::Status)> f) {
  ::grpc::internal::CallbackUnaryCall< ::viam::service::sensors::v1::GetReadingsRequest, ::viam::service::sensors::v1::GetReadingsResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_GetReadings_, context, request, response, std::move(f));
}

void SensorsService::Stub::async::GetReadings(::grpc::ClientContext* context, const ::viam::service::sensors::v1::GetReadingsRequest* request, ::viam::service::sensors::v1::GetReadingsResponse* response, ::grpc::ClientUnaryReactor* reactor) {
  ::grpc::internal::ClientCallbackUnaryFactory::Create< ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_GetReadings_, context, request, response, reactor);
}

::grpc::ClientAsyncResponseReader< ::viam::service::sensors::v1::GetReadingsResponse>* SensorsService::Stub::PrepareAsyncGetReadingsRaw(::grpc::ClientContext* context, const ::viam::service::sensors::v1::GetReadingsRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderHelper::Create< ::viam::service::sensors::v1::GetReadingsResponse, ::viam::service::sensors::v1::GetReadingsRequest, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), cq, rpcmethod_GetReadings_, context, request);
}

::grpc::ClientAsyncResponseReader< ::viam::service::sensors::v1::GetReadingsResponse>* SensorsService::Stub::AsyncGetReadingsRaw(::grpc::ClientContext* context, const ::viam::service::sensors::v1::GetReadingsRequest& request, ::grpc::CompletionQueue* cq) {
  auto* result =
    this->PrepareAsyncGetReadingsRaw(context, request, cq);
  result->StartCall();
  return result;
}

::grpc::Status SensorsService::Stub::DoCommand(::grpc::ClientContext* context, const ::viam::common::v1::DoCommandRequest& request, ::viam::common::v1::DoCommandResponse* response) {
  return ::grpc::internal::BlockingUnaryCall< ::viam::common::v1::DoCommandRequest, ::viam::common::v1::DoCommandResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), rpcmethod_DoCommand_, context, request, response);
}

void SensorsService::Stub::async::DoCommand(::grpc::ClientContext* context, const ::viam::common::v1::DoCommandRequest* request, ::viam::common::v1::DoCommandResponse* response, std::function<void(::grpc::Status)> f) {
  ::grpc::internal::CallbackUnaryCall< ::viam::common::v1::DoCommandRequest, ::viam::common::v1::DoCommandResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_DoCommand_, context, request, response, std::move(f));
}

void SensorsService::Stub::async::DoCommand(::grpc::ClientContext* context, const ::viam::common::v1::DoCommandRequest* request, ::viam::common::v1::DoCommandResponse* response, ::grpc::ClientUnaryReactor* reactor) {
  ::grpc::internal::ClientCallbackUnaryFactory::Create< ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_DoCommand_, context, request, response, reactor);
}

::grpc::ClientAsyncResponseReader< ::viam::common::v1::DoCommandResponse>* SensorsService::Stub::PrepareAsyncDoCommandRaw(::grpc::ClientContext* context, const ::viam::common::v1::DoCommandRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderHelper::Create< ::viam::common::v1::DoCommandResponse, ::viam::common::v1::DoCommandRequest, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), cq, rpcmethod_DoCommand_, context, request);
}

::grpc::ClientAsyncResponseReader< ::viam::common::v1::DoCommandResponse>* SensorsService::Stub::AsyncDoCommandRaw(::grpc::ClientContext* context, const ::viam::common::v1::DoCommandRequest& request, ::grpc::CompletionQueue* cq) {
  auto* result =
    this->PrepareAsyncDoCommandRaw(context, request, cq);
  result->StartCall();
  return result;
}

SensorsService::Service::Service() {
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      SensorsService_method_names[0],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< SensorsService::Service, ::viam::service::sensors::v1::GetSensorsRequest, ::viam::service::sensors::v1::GetSensorsResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(
          [](SensorsService::Service* service,
             ::grpc::ServerContext* ctx,
             const ::viam::service::sensors::v1::GetSensorsRequest* req,
             ::viam::service::sensors::v1::GetSensorsResponse* resp) {
               return service->GetSensors(ctx, req, resp);
             }, this)));
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      SensorsService_method_names[1],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< SensorsService::Service, ::viam::service::sensors::v1::GetReadingsRequest, ::viam::service::sensors::v1::GetReadingsResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(
          [](SensorsService::Service* service,
             ::grpc::ServerContext* ctx,
             const ::viam::service::sensors::v1::GetReadingsRequest* req,
             ::viam::service::sensors::v1::GetReadingsResponse* resp) {
               return service->GetReadings(ctx, req, resp);
             }, this)));
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      SensorsService_method_names[2],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< SensorsService::Service, ::viam::common::v1::DoCommandRequest, ::viam::common::v1::DoCommandResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(
          [](SensorsService::Service* service,
             ::grpc::ServerContext* ctx,
             const ::viam::common::v1::DoCommandRequest* req,
             ::viam::common::v1::DoCommandResponse* resp) {
               return service->DoCommand(ctx, req, resp);
             }, this)));
}

SensorsService::Service::~Service() {
}

::grpc::Status SensorsService::Service::GetSensors(::grpc::ServerContext* context, const ::viam::service::sensors::v1::GetSensorsRequest* request, ::viam::service::sensors::v1::GetSensorsResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status SensorsService::Service::GetReadings(::grpc::ServerContext* context, const ::viam::service::sensors::v1::GetReadingsRequest* request, ::viam::service::sensors::v1::GetReadingsResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status SensorsService::Service::DoCommand(::grpc::ServerContext* context, const ::viam::common::v1::DoCommandRequest* request, ::viam::common::v1::DoCommandResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}


}  // namespace viam
}  // namespace service
}  // namespace sensors
}  // namespace v1

