# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/ray/protobuf/export_api/export_actor_data.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import common_pb2 as src_dot_ray_dot_protobuf_dot_common__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n3src/ray/protobuf/export_api/export_actor_data.proto\x12\x07ray.rpc\x1a\x1dsrc/ray/protobuf/common.proto\"\x94\x06\n\x0f\x45xportActorData\x12\x19\n\x08\x61\x63tor_id\x18\x01 \x01(\x0cR\x07\x61\x63torId\x12\x15\n\x06job_id\x18\x02 \x01(\x0cR\x05jobId\x12\x39\n\x05state\x18\x03 \x01(\x0e\x32#.ray.rpc.ExportActorData.ActorStateR\x05state\x12\x1f\n\x0bis_detached\x18\x04 \x01(\x08R\nisDetached\x12\x12\n\x04name\x18\x05 \x01(\tR\x04name\x12\x10\n\x03pid\x18\x06 \x01(\rR\x03pid\x12#\n\rray_namespace\x18\x07 \x01(\tR\x0crayNamespace\x12\x34\n\x16serialized_runtime_env\x18\x08 \x01(\tR\x14serializedRuntimeEnv\x12\x1d\n\nclass_name\x18\t \x01(\tR\tclassName\x12\x39\n\x0b\x64\x65\x61th_cause\x18\n \x01(\x0b\x32\x18.ray.rpc.ActorDeathCauseR\ndeathCause\x12^\n\x12required_resources\x18\x0b \x03(\x0b\x32/.ray.rpc.ExportActorData.RequiredResourcesEntryR\x11requiredResources\x12\x1c\n\x07node_id\x18\x0c \x01(\x0cH\x00R\x06nodeId\x88\x01\x01\x12\x31\n\x12placement_group_id\x18\r \x01(\x0cH\x01R\x10placementGroupId\x88\x01\x01\x12\x1b\n\trepr_name\x18\x0e \x01(\tR\x08reprName\x1a\x44\n\x16RequiredResourcesEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\x01R\x05value:\x02\x38\x01\"a\n\nActorState\x12\x18\n\x14\x44\x45PENDENCIES_UNREADY\x10\x00\x12\x14\n\x10PENDING_CREATION\x10\x01\x12\t\n\x05\x41LIVE\x10\x02\x12\x0e\n\nRESTARTING\x10\x03\x12\x08\n\x04\x44\x45\x41\x44\x10\x04\x42\n\n\x08_node_idB\x15\n\x13_placement_group_idB\x03\xf8\x01\x01\x62\x06proto3')



_EXPORTACTORDATA = DESCRIPTOR.message_types_by_name['ExportActorData']
_EXPORTACTORDATA_REQUIREDRESOURCESENTRY = _EXPORTACTORDATA.nested_types_by_name['RequiredResourcesEntry']
_EXPORTACTORDATA_ACTORSTATE = _EXPORTACTORDATA.enum_types_by_name['ActorState']
ExportActorData = _reflection.GeneratedProtocolMessageType('ExportActorData', (_message.Message,), {

  'RequiredResourcesEntry' : _reflection.GeneratedProtocolMessageType('RequiredResourcesEntry', (_message.Message,), {
    'DESCRIPTOR' : _EXPORTACTORDATA_REQUIREDRESOURCESENTRY,
    '__module__' : 'src.ray.protobuf.export_api.export_actor_data_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.ExportActorData.RequiredResourcesEntry)
    })
  ,
  'DESCRIPTOR' : _EXPORTACTORDATA,
  '__module__' : 'src.ray.protobuf.export_api.export_actor_data_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ExportActorData)
  })
_sym_db.RegisterMessage(ExportActorData)
_sym_db.RegisterMessage(ExportActorData.RequiredResourcesEntry)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\370\001\001'
  _EXPORTACTORDATA_REQUIREDRESOURCESENTRY._options = None
  _EXPORTACTORDATA_REQUIREDRESOURCESENTRY._serialized_options = b'8\001'
  _EXPORTACTORDATA._serialized_start=96
  _EXPORTACTORDATA._serialized_end=884
  _EXPORTACTORDATA_REQUIREDRESOURCESENTRY._serialized_start=682
  _EXPORTACTORDATA_REQUIREDRESOURCESENTRY._serialized_end=750
  _EXPORTACTORDATA_ACTORSTATE._serialized_start=752
  _EXPORTACTORDATA_ACTORSTATE._serialized_end=849
# @@protoc_insertion_point(module_scope)
