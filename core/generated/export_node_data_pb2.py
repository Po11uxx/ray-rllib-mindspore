# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/ray/protobuf/export_api/export_node_data.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n2src/ray/protobuf/export_api/export_node_data.proto\x12\x07ray.rpc\"\xa2\x07\n\x0e\x45xportNodeData\x12\x17\n\x07node_id\x18\x01 \x01(\x0cR\x06nodeId\x12\x30\n\x14node_manager_address\x18\x02 \x01(\tR\x12nodeManagerAddress\x12:\n\x05state\x18\x03 \x01(\x0e\x32$.ray.rpc.ExportNodeData.GcsNodeStateR\x05state\x12T\n\x0fresources_total\x18\x04 \x03(\x0b\x32+.ray.rpc.ExportNodeData.ResourcesTotalEntryR\x0eresourcesTotal\x12\x1b\n\tnode_name\x18\x05 \x01(\tR\x08nodeName\x12\"\n\rstart_time_ms\x18\x06 \x01(\x04R\x0bstartTimeMs\x12\x1e\n\x0b\x65nd_time_ms\x18\x07 \x01(\x04R\tendTimeMs\x12 \n\x0cis_head_node\x18\x08 \x01(\x08R\nisHeadNode\x12;\n\x06labels\x18\t \x03(\x0b\x32#.ray.rpc.ExportNodeData.LabelsEntryR\x06labels\x12\x44\n\ndeath_info\x18\n \x01(\x0b\x32%.ray.rpc.ExportNodeData.NodeDeathInfoR\tdeathInfo\x1a\x89\x02\n\rNodeDeathInfo\x12\x44\n\x06reason\x18\x01 \x01(\x0e\x32,.ray.rpc.ExportNodeData.NodeDeathInfo.ReasonR\x06reason\x12%\n\x0ereason_message\x18\x02 \x01(\tR\rreasonMessage\"\x8a\x01\n\x06Reason\x12\x0f\n\x0bUNSPECIFIED\x10\x00\x12\x18\n\x14\x45XPECTED_TERMINATION\x10\x01\x12\x1a\n\x16UNEXPECTED_TERMINATION\x10\x02\x12\x1e\n\x1a\x41UTOSCALER_DRAIN_PREEMPTED\x10\x03\x12\x19\n\x15\x41UTOSCALER_DRAIN_IDLE\x10\x04\x1a\x41\n\x13ResourcesTotalEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\x01R\x05value:\x02\x38\x01\x1a\x39\n\x0bLabelsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value:\x02\x38\x01\"#\n\x0cGcsNodeState\x12\t\n\x05\x41LIVE\x10\x00\x12\x08\n\x04\x44\x45\x41\x44\x10\x01\x42\x03\xf8\x01\x01\x62\x06proto3')



_EXPORTNODEDATA = DESCRIPTOR.message_types_by_name['ExportNodeData']
_EXPORTNODEDATA_NODEDEATHINFO = _EXPORTNODEDATA.nested_types_by_name['NodeDeathInfo']
_EXPORTNODEDATA_RESOURCESTOTALENTRY = _EXPORTNODEDATA.nested_types_by_name['ResourcesTotalEntry']
_EXPORTNODEDATA_LABELSENTRY = _EXPORTNODEDATA.nested_types_by_name['LabelsEntry']
_EXPORTNODEDATA_NODEDEATHINFO_REASON = _EXPORTNODEDATA_NODEDEATHINFO.enum_types_by_name['Reason']
_EXPORTNODEDATA_GCSNODESTATE = _EXPORTNODEDATA.enum_types_by_name['GcsNodeState']
ExportNodeData = _reflection.GeneratedProtocolMessageType('ExportNodeData', (_message.Message,), {

  'NodeDeathInfo' : _reflection.GeneratedProtocolMessageType('NodeDeathInfo', (_message.Message,), {
    'DESCRIPTOR' : _EXPORTNODEDATA_NODEDEATHINFO,
    '__module__' : 'src.ray.protobuf.export_api.export_node_data_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.ExportNodeData.NodeDeathInfo)
    })
  ,

  'ResourcesTotalEntry' : _reflection.GeneratedProtocolMessageType('ResourcesTotalEntry', (_message.Message,), {
    'DESCRIPTOR' : _EXPORTNODEDATA_RESOURCESTOTALENTRY,
    '__module__' : 'src.ray.protobuf.export_api.export_node_data_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.ExportNodeData.ResourcesTotalEntry)
    })
  ,

  'LabelsEntry' : _reflection.GeneratedProtocolMessageType('LabelsEntry', (_message.Message,), {
    'DESCRIPTOR' : _EXPORTNODEDATA_LABELSENTRY,
    '__module__' : 'src.ray.protobuf.export_api.export_node_data_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.ExportNodeData.LabelsEntry)
    })
  ,
  'DESCRIPTOR' : _EXPORTNODEDATA,
  '__module__' : 'src.ray.protobuf.export_api.export_node_data_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ExportNodeData)
  })
_sym_db.RegisterMessage(ExportNodeData)
_sym_db.RegisterMessage(ExportNodeData.NodeDeathInfo)
_sym_db.RegisterMessage(ExportNodeData.ResourcesTotalEntry)
_sym_db.RegisterMessage(ExportNodeData.LabelsEntry)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\370\001\001'
  _EXPORTNODEDATA_RESOURCESTOTALENTRY._options = None
  _EXPORTNODEDATA_RESOURCESTOTALENTRY._serialized_options = b'8\001'
  _EXPORTNODEDATA_LABELSENTRY._options = None
  _EXPORTNODEDATA_LABELSENTRY._serialized_options = b'8\001'
  _EXPORTNODEDATA._serialized_start=64
  _EXPORTNODEDATA._serialized_end=994
  _EXPORTNODEDATA_NODEDEATHINFO._serialized_start=566
  _EXPORTNODEDATA_NODEDEATHINFO._serialized_end=831
  _EXPORTNODEDATA_NODEDEATHINFO_REASON._serialized_start=693
  _EXPORTNODEDATA_NODEDEATHINFO_REASON._serialized_end=831
  _EXPORTNODEDATA_RESOURCESTOTALENTRY._serialized_start=833
  _EXPORTNODEDATA_RESOURCESTOTALENTRY._serialized_end=898
  _EXPORTNODEDATA_LABELSENTRY._serialized_start=900
  _EXPORTNODEDATA_LABELSENTRY._serialized_end=957
  _EXPORTNODEDATA_GCSNODESTATE._serialized_start=959
  _EXPORTNODEDATA_GCSNODESTATE._serialized_end=994
# @@protoc_insertion_point(module_scope)
