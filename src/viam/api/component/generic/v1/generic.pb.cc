// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: component/generic/v1/generic.proto

#include "component/generic/v1/generic.pb.h"

#include <algorithm>
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/extension_set.h"
#include "google/protobuf/wire_format_lite.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/generated_message_reflection.h"
#include "google/protobuf/reflection_ops.h"
#include "google/protobuf/wire_format.h"
#include "google/protobuf/generated_message_tctable_impl.h"
// @@protoc_insertion_point(includes)

// Must be included last.
#include "google/protobuf/port_def.inc"
PROTOBUF_PRAGMA_INIT_SEG
namespace _pb = ::google::protobuf;
namespace _pbi = ::google::protobuf::internal;
namespace _fl = ::google::protobuf::internal::field_layout;
namespace viam {
namespace component {
namespace generic {
namespace v1 {
}  // namespace v1
}  // namespace generic
}  // namespace component
}  // namespace viam
static constexpr const ::_pb::EnumDescriptor**
    file_level_enum_descriptors_component_2fgeneric_2fv1_2fgeneric_2eproto = nullptr;
static constexpr const ::_pb::ServiceDescriptor**
    file_level_service_descriptors_component_2fgeneric_2fv1_2fgeneric_2eproto = nullptr;
const ::uint32_t TableStruct_component_2fgeneric_2fv1_2fgeneric_2eproto::offsets[1] = {};
static constexpr ::_pbi::MigrationSchema* schemas = nullptr;
static constexpr ::_pb::Message* const* file_default_instances = nullptr;
const char descriptor_table_protodef_component_2fgeneric_2fv1_2fgeneric_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
    "\n\"component/generic/v1/generic.proto\022\031vi"
    "am.component.generic.v1\032\026common/v1/commo"
    "n.proto\032\034google/api/annotations.proto2\266\002"
    "\n\016GenericService\022\212\001\n\tDoCommand\022 .viam.co"
    "mmon.v1.DoCommandRequest\032!.viam.common.v"
    "1.DoCommandResponse\"8\202\323\344\223\0022\"0/viam/api/v"
    "1/component/generic/{name}/do_command\022\226\001"
    "\n\rGetGeometries\022$.viam.common.v1.GetGeom"
    "etriesRequest\032%.viam.common.v1.GetGeomet"
    "riesResponse\"8\202\323\344\223\0022\0220/viam/api/v1/compo"
    "nent/generic/{name}/geometriesB\332\001\n\035com.v"
    "iam.component.generic.v1B\014GenericProtoP\001"
    "Z$go.viam.com/api/component/generic/v1\242\002"
    "\003VCG\252\002\031Viam.Component.Generic.V1\312\002\031Viam\\"
    "Component\\Generic\\V1\342\002%Viam\\Component\\Ge"
    "neric\\V1\\GPBMetadata\352\002\034Viam::Component::"
    "Generic::V1b\006proto3"
};
static const ::_pbi::DescriptorTable* const descriptor_table_component_2fgeneric_2fv1_2fgeneric_2eproto_deps[2] =
    {
        &::descriptor_table_common_2fv1_2fcommon_2eproto,
        &::descriptor_table_google_2fapi_2fannotations_2eproto,
};
static ::absl::once_flag descriptor_table_component_2fgeneric_2fv1_2fgeneric_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_component_2fgeneric_2fv1_2fgeneric_2eproto = {
    false,
    false,
    659,
    descriptor_table_protodef_component_2fgeneric_2fv1_2fgeneric_2eproto,
    "component/generic/v1/generic.proto",
    &descriptor_table_component_2fgeneric_2fv1_2fgeneric_2eproto_once,
    descriptor_table_component_2fgeneric_2fv1_2fgeneric_2eproto_deps,
    2,
    0,
    schemas,
    file_default_instances,
    TableStruct_component_2fgeneric_2fv1_2fgeneric_2eproto::offsets,
    nullptr,
    file_level_enum_descriptors_component_2fgeneric_2fv1_2fgeneric_2eproto,
    file_level_service_descriptors_component_2fgeneric_2fv1_2fgeneric_2eproto,
};

// This function exists to be marked as weak.
// It can significantly speed up compilation by breaking up LLVM's SCC
// in the .pb.cc translation units. Large translation units see a
// reduction of more than 35% of walltime for optimized builds. Without
// the weak attribute all the messages in the file, including all the
// vtables and everything they use become part of the same SCC through
// a cycle like:
// GetMetadata -> descriptor table -> default instances ->
//   vtables -> GetMetadata
// By adding a weak function here we break the connection from the
// individual vtables back into the descriptor table.
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_component_2fgeneric_2fv1_2fgeneric_2eproto_getter() {
  return &descriptor_table_component_2fgeneric_2fv1_2fgeneric_2eproto;
}
// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2
static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_component_2fgeneric_2fv1_2fgeneric_2eproto(&descriptor_table_component_2fgeneric_2fv1_2fgeneric_2eproto);
namespace viam {
namespace component {
namespace generic {
namespace v1 {
// @@protoc_insertion_point(namespace_scope)
}  // namespace v1
}  // namespace generic
}  // namespace component
}  // namespace viam
namespace google {
namespace protobuf {
}  // namespace protobuf
}  // namespace google
// @@protoc_insertion_point(global_scope)
#include "google/protobuf/port_undef.inc"
