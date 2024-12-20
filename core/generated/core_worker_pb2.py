# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/ray/protobuf/core_worker.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import common_pb2 as src_dot_ray_dot_protobuf_dot_common__pb2
from . import pubsub_pb2 as src_dot_ray_dot_protobuf_dot_pubsub__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\"src/ray/protobuf/core_worker.proto\x12\x07ray.rpc\x1a\x1dsrc/ray/protobuf/common.proto\x1a\x1dsrc/ray/protobuf/pubsub.proto\"0\n\x0f\x41\x63tiveObjectIDs\x12\x1d\n\nobject_ids\x18\x01 \x03(\x0cR\tobjectIds\"\x87\x05\n\x0b\x41\x63torHandle\x12\x19\n\x08\x61\x63tor_id\x18\x01 \x01(\x0cR\x07\x61\x63torId\x12\x19\n\x08owner_id\x18\x02 \x01(\x0cR\x07ownerId\x12\x35\n\rowner_address\x18\x03 \x01(\x0b\x32\x10.ray.rpc.AddressR\x0cownerAddress\x12&\n\x0f\x63reation_job_id\x18\x04 \x01(\x0cR\rcreationJobId\x12\x38\n\x0e\x61\x63tor_language\x18\x05 \x01(\x0e\x32\x11.ray.rpc.LanguageR\ractorLanguage\x12q\n\'actor_creation_task_function_descriptor\x18\x06 \x01(\x0b\x32\x1b.ray.rpc.FunctionDescriptorR#actorCreationTaskFunctionDescriptor\x12!\n\x0c\x61\x63tor_cursor\x18\x07 \x01(\x0cR\x0b\x61\x63torCursor\x12%\n\x0e\x65xtension_data\x18\x08 \x01(\x0cR\rextensionData\x12(\n\x10max_task_retries\x18\t \x01(\x03R\x0emaxTaskRetries\x12\x12\n\x04name\x18\n \x01(\tR\x04name\x12#\n\rray_namespace\x18\x0b \x01(\tR\x0crayNamespace\x12/\n\x14\x65xecute_out_of_order\x18\x0c \x01(\x08R\x11\x65xecuteOutOfOrder\x12*\n\x11max_pending_calls\x18\r \x01(\x05R\x0fmaxPendingCalls\x12,\n\x12\x65nable_task_events\x18\x0e \x01(\x08R\x10\x65nableTaskEvents\"\x93\x02\n\x0fPushTaskRequest\x12,\n\x12intended_worker_id\x18\x01 \x01(\x0cR\x10intendedWorkerId\x12.\n\ttask_spec\x18\x02 \x01(\x0b\x32\x11.ray.rpc.TaskSpecR\x08taskSpec\x12\'\n\x0fsequence_number\x18\x03 \x01(\x03R\x0esequenceNumber\x12\x33\n\x16\x63lient_processed_up_to\x18\x04 \x01(\x03R\x13\x63lientProcessedUpTo\x12\x44\n\x10resource_mapping\x18\x05 \x03(\x0b\x32\x19.ray.rpc.ResourceMapEntryR\x0fresourceMapping\"\x87\x05\n\rPushTaskReply\x12<\n\x0ereturn_objects\x18\x01 \x03(\x0b\x32\x15.ray.rpc.ReturnObjectR\rreturnObjects\x12K\n\x16\x64ynamic_return_objects\x18\x02 \x03(\x0b\x32\x15.ray.rpc.ReturnObjectR\x14\x64ynamicReturnObjects\x12%\n\x0eworker_exiting\x18\x03 \x01(\x08R\rworkerExiting\x12\x42\n\rborrowed_refs\x18\x04 \x03(\x0b\x32\x1d.ray.rpc.ObjectReferenceCountR\x0c\x62orrowedRefs\x12,\n\x12is_retryable_error\x18\x05 \x01(\x08R\x10isRetryableError\x12\x30\n\x14is_application_error\x18\x06 \x01(\x08R\x12isApplicationError\x12?\n\x1cwas_cancelled_before_running\x18\x07 \x01(\x08R\x19wasCancelledBeforeRunning\x12+\n\x0f\x61\x63tor_repr_name\x18\x08 \x01(\tH\x00R\ractorReprName\x88\x01\x01\x12\x30\n\x14task_execution_error\x18\t \x01(\tR\x12taskExecutionError\x12l\n\x1estreaming_generator_return_ids\x18\n \x03(\x0b\x32\'.ray.rpc.StreamingGeneratorReturnIdInfoR\x1bstreamingGeneratorReturnIdsB\x12\n\x10_actor_repr_name\"g\n%DirectActorCallArgWaitCompleteRequest\x12,\n\x12intended_worker_id\x18\x01 \x01(\x0cR\x10intendedWorkerId\x12\x10\n\x03tag\x18\x02 \x01(\x03R\x03tag\"%\n#DirectActorCallArgWaitCompleteReply\"]\n\x16GetObjectStatusRequest\x12&\n\x0fowner_worker_id\x18\x01 \x01(\x0cR\rownerWorkerId\x12\x1b\n\tobject_id\x18\x02 \x01(\x0cR\x08objectId\"\x85\x01\n\tRayObject\x12\x12\n\x04\x64\x61ta\x18\x01 \x01(\x0cR\x04\x64\x61ta\x12\x1a\n\x08metadata\x18\x02 \x01(\x0cR\x08metadata\x12H\n\x13nested_inlined_refs\x18\x03 \x03(\x0b\x32\x18.ray.rpc.ObjectReferenceR\x11nestedInlinedRefs\"\xfc\x01\n\x14GetObjectStatusReply\x12\x42\n\x06status\x18\x01 \x01(\x0e\x32*.ray.rpc.GetObjectStatusReply.ObjectStatusR\x06status\x12*\n\x06object\x18\x02 \x01(\x0b\x32\x12.ray.rpc.RayObjectR\x06object\x12\x19\n\x08node_ids\x18\x03 \x03(\x0cR\x07nodeIds\x12\x1f\n\x0bobject_size\x18\x04 \x01(\x04R\nobjectSize\"8\n\x0cObjectStatus\x12\x0b\n\x07\x43REATED\x10\x00\x12\x10\n\x0cOUT_OF_SCOPE\x10\x01\x12\t\n\x05\x46REED\x10\x02\"h\n\x1dWaitForActorRefDeletedRequest\x12,\n\x12intended_worker_id\x18\x01 \x01(\x0cR\x10intendedWorkerId\x12\x19\n\x08\x61\x63tor_id\x18\x02 \x01(\x0cR\x07\x61\x63torId\"\x1d\n\x1bWaitForActorRefDeletedReply\"\xc0\x01\n UpdateObjectLocationBatchRequest\x12,\n\x12intended_worker_id\x18\x01 \x01(\x0cR\x10intendedWorkerId\x12\x17\n\x07node_id\x18\x02 \x01(\x0cR\x06nodeId\x12U\n\x17object_location_updates\x18\x03 \x03(\x0b\x32\x1d.ray.rpc.ObjectLocationUpdateR\x15objectLocationUpdates\" \n\x1eUpdateObjectLocationBatchReply\"w\n\x1bObjectSpilledLocationUpdate\x12\x1f\n\x0bspilled_url\x18\x03 \x01(\tR\nspilledUrl\x12\x37\n\x18spilled_to_local_storage\x18\x04 \x01(\x08R\x15spilledToLocalStorage\"\xe6\x02\n\x14ObjectLocationUpdate\x12\x1b\n\tobject_id\x18\x01 \x01(\x0cR\x08objectId\x12^\n\x16plasma_location_update\x18\x02 \x01(\x0e\x32#.ray.rpc.ObjectPlasmaLocationUpdateH\x00R\x14plasmaLocationUpdate\x88\x01\x01\x12\x61\n\x17spilled_location_update\x18\x03 \x01(\x0b\x32$.ray.rpc.ObjectSpilledLocationUpdateH\x01R\x15spilledLocationUpdate\x88\x01\x01\x12&\n\x0cgenerator_id\x18\x04 \x01(\x0cH\x02R\x0bgeneratorId\x88\x01\x01\x42\x19\n\x17_plasma_location_updateB\x1a\n\x18_spilled_location_updateB\x0f\n\r_generator_id\"m\n\x1eGetObjectLocationsOwnerRequest\x12,\n\x12intended_worker_id\x18\x01 \x01(\x0cR\x10intendedWorkerId\x12\x1d\n\nobject_ids\x18\x02 \x03(\x0cR\tobjectIds\"|\n\x1cGetObjectLocationsOwnerReply\x12\\\n\x15object_location_infos\x18\x01 \x03(\x0b\x32(.ray.rpc.WorkerObjectLocationsPubMessageR\x13objectLocationInfos\"\x98\x01\n\x10KillActorRequest\x12*\n\x11intended_actor_id\x18\x01 \x01(\x0cR\x0fintendedActorId\x12\x1d\n\nforce_kill\x18\x02 \x01(\x08R\tforceKill\x12\x39\n\x0b\x64\x65\x61th_cause\x18\x03 \x01(\x0b\x32\x18.ray.rpc.ActorDeathCauseR\ndeathCause\"\x10\n\x0eKillActorReply\"\xa4\x01\n\x11\x43\x61ncelTaskRequest\x12(\n\x10intended_task_id\x18\x01 \x01(\x0cR\x0eintendedTaskId\x12\x1d\n\nforce_kill\x18\x02 \x01(\x08R\tforceKill\x12\x1c\n\trecursive\x18\x03 \x01(\x08R\trecursive\x12(\n\x10\x63\x61ller_worker_id\x18\x04 \x01(\x0cR\x0e\x63\x61llerWorkerId\"t\n\x0f\x43\x61ncelTaskReply\x12\x34\n\x16requested_task_running\x18\x01 \x01(\x08R\x14requestedTaskRunning\x12+\n\x11\x61ttempt_succeeded\x18\x02 \x01(\x08R\x10\x61ttemptSucceeded\"\x80\x01\n\x17RemoteCancelTaskRequest\x12(\n\x10remote_object_id\x18\x01 \x01(\x0cR\x0eremoteObjectId\x12\x1d\n\nforce_kill\x18\x02 \x01(\x08R\tforceKill\x12\x1c\n\trecursive\x18\x03 \x01(\x08R\trecursive\"\x17\n\x15RemoteCancelTaskReply\"\xca\x01\n\x19GetCoreWorkerStatsRequest\x12,\n\x12intended_worker_id\x18\x01 \x01(\x0cR\x10intendedWorkerId\x12.\n\x13include_memory_info\x18\x02 \x01(\x08R\x11includeMemoryInfo\x12*\n\x11include_task_info\x18\x03 \x01(\x08R\x0fincludeTaskInfo\x12\x19\n\x05limit\x18\x04 \x01(\x03H\x00R\x05limit\x88\x01\x01\x42\x08\n\x06_limit\"\xf9\x01\n\x17GetCoreWorkerStatsReply\x12\x44\n\x11\x63ore_worker_stats\x18\x01 \x01(\x0b\x32\x18.ray.rpc.CoreWorkerStatsR\x0f\x63oreWorkerStats\x12M\n\x17owned_task_info_entries\x18\x02 \x03(\x0b\x32\x16.ray.rpc.TaskInfoEntryR\x14ownedTaskInfoEntries\x12(\n\x10running_task_ids\x18\x03 \x03(\x0cR\x0erunningTaskIds\x12\x1f\n\x0btasks_total\x18\x04 \x01(\x03R\ntasksTotal\"E\n\x0eLocalGCRequest\x12\x33\n\x16triggered_by_global_gc\x18\x01 \x01(\x08R\x13triggeredByGlobalGc\"\x0e\n\x0cLocalGCReply\"7\n\x18PlasmaObjectReadyRequest\x12\x1b\n\tobject_id\x18\x01 \x01(\x0cR\x08objectId\"\x18\n\x16PlasmaObjectReadyReply\"T\n\x14\x44\x65leteObjectsRequest\x12\x1d\n\nobject_ids\x18\x01 \x03(\x0cR\tobjectIds\x12\x1d\n\nlocal_only\x18\x02 \x01(\x08R\tlocalOnly\"\x14\n\x12\x44\x65leteObjectsReply\"\xa6\x01\n\x13SpillObjectsRequest\x12I\n\x14object_refs_to_spill\x18\x01 \x03(\x0b\x32\x18.ray.rpc.ObjectReferenceR\x11objectRefsToSpill\x12\x44\n\x0e\x64\x65lete_request\x18\x02 \x01(\x0b\x32\x1d.ray.rpc.DeleteObjectsRequestR\rdeleteRequest\"C\n\x11SpillObjectsReply\x12.\n\x13spilled_objects_url\x18\x01 \x03(\tR\x11spilledObjectsUrl\"\x81\x01\n\x1cRestoreSpilledObjectsRequest\x12.\n\x13spilled_objects_url\x18\x01 \x03(\tR\x11spilledObjectsUrl\x12\x31\n\x15object_ids_to_restore\x18\x02 \x03(\x0cR\x12objectIdsToRestore\"N\n\x1aRestoreSpilledObjectsReply\x12\x30\n\x14\x62ytes_restored_total\x18\x01 \x01(\x03R\x12\x62ytesRestoredTotal\"M\n\x1b\x44\x65leteSpilledObjectsRequest\x12.\n\x13spilled_objects_url\x18\x01 \x03(\tR\x11spilledObjectsUrl\"\x1b\n\x19\x44\x65leteSpilledObjectsReply\",\n\x0b\x45xitRequest\x12\x1d\n\nforce_exit\x18\x01 \x01(\x08R\tforceExit\"%\n\tExitReply\x12\x18\n\x07success\x18\x01 \x01(\x08R\x07success\"\xe4\x01\n\x18\x41ssignObjectOwnerRequest\x12\x1b\n\tobject_id\x18\x01 \x01(\x0cR\x08objectId\x12\x1f\n\x0bobject_size\x18\x02 \x01(\x04R\nobjectSize\x12\x30\n\x14\x63ontained_object_ids\x18\x03 \x03(\x0cR\x12\x63ontainedObjectIds\x12;\n\x10\x62orrower_address\x18\x04 \x01(\x0b\x32\x10.ray.rpc.AddressR\x0f\x62orrowerAddress\x12\x1b\n\tcall_site\x18\x05 \x01(\tR\x08\x63\x61llSite\"\x18\n\x16\x41ssignObjectOwnerReply\"\x1f\n\x1dRayletNotifyGCSRestartRequest\"\x1d\n\x1bRayletNotifyGCSRestartReply\"\x18\n\x16NumPendingTasksRequest\"B\n\x14NumPendingTasksReply\x12*\n\x11num_pending_tasks\x18\x01 \x01(\x03R\x0fnumPendingTasks\"\x8c\x02\n!ReportGeneratorItemReturnsRequest\x12K\n\x16\x64ynamic_return_objects\x18\x01 \x03(\x0b\x32\x15.ray.rpc.ReturnObjectR\x14\x64ynamicReturnObjects\x12\x31\n\x0bworker_addr\x18\x02 \x01(\x0b\x32\x10.ray.rpc.AddressR\nworkerAddr\x12\x1d\n\nitem_index\x18\x03 \x01(\x03R\titemIndex\x12!\n\x0cgenerator_id\x18\x05 \x01(\x0cR\x0bgeneratorId\x12%\n\x0e\x61ttempt_number\x18\x06 \x01(\x04R\rattemptNumber\"\\\n\x1fReportGeneratorItemReturnsReply\x12\x39\n\x19total_num_object_consumed\x18\x01 \x01(\x03R\x16totalNumObjectConsumed\"\x99\x01\n\"RegisterMutableObjectReaderRequest\x12(\n\x10writer_object_id\x18\x01 \x01(\x0cR\x0ewriterObjectId\x12\x1f\n\x0bnum_readers\x18\x02 \x01(\x03R\nnumReaders\x12(\n\x10reader_object_id\x18\x03 \x01(\x0cR\x0ereaderObjectId\"\"\n RegisterMutableObjectReaderReply*4\n\x1aObjectPlasmaLocationUpdate\x12\t\n\x05\x41\x44\x44\x45\x44\x10\x00\x12\x0b\n\x07REMOVED\x10\x01\x32\xf7\x10\n\x11\x43oreWorkerService\x12\x66\n\x16RayletNotifyGCSRestart\x12&.ray.rpc.RayletNotifyGCSRestartRequest\x1a$.ray.rpc.RayletNotifyGCSRestartReply\x12<\n\x08PushTask\x12\x18.ray.rpc.PushTaskRequest\x1a\x16.ray.rpc.PushTaskReply\x12~\n\x1e\x44irectActorCallArgWaitComplete\x12..ray.rpc.DirectActorCallArgWaitCompleteRequest\x1a,.ray.rpc.DirectActorCallArgWaitCompleteReply\x12Q\n\x0fGetObjectStatus\x12\x1f.ray.rpc.GetObjectStatusRequest\x1a\x1d.ray.rpc.GetObjectStatusReply\x12\x66\n\x16WaitForActorRefDeleted\x12&.ray.rpc.WaitForActorRefDeletedRequest\x1a$.ray.rpc.WaitForActorRefDeletedReply\x12W\n\x11PubsubLongPolling\x12!.ray.rpc.PubsubLongPollingRequest\x1a\x1f.ray.rpc.PubsubLongPollingReply\x12r\n\x1aReportGeneratorItemReturns\x12*.ray.rpc.ReportGeneratorItemReturnsRequest\x1a(.ray.rpc.ReportGeneratorItemReturnsReply\x12Z\n\x12PubsubCommandBatch\x12\".ray.rpc.PubsubCommandBatchRequest\x1a .ray.rpc.PubsubCommandBatchReply\x12o\n\x19UpdateObjectLocationBatch\x12).ray.rpc.UpdateObjectLocationBatchRequest\x1a\'.ray.rpc.UpdateObjectLocationBatchReply\x12i\n\x17GetObjectLocationsOwner\x12\'.ray.rpc.GetObjectLocationsOwnerRequest\x1a%.ray.rpc.GetObjectLocationsOwnerReply\x12?\n\tKillActor\x12\x19.ray.rpc.KillActorRequest\x1a\x17.ray.rpc.KillActorReply\x12\x42\n\nCancelTask\x12\x1a.ray.rpc.CancelTaskRequest\x1a\x18.ray.rpc.CancelTaskReply\x12T\n\x10RemoteCancelTask\x12 .ray.rpc.RemoteCancelTaskRequest\x1a\x1e.ray.rpc.RemoteCancelTaskReply\x12Z\n\x12GetCoreWorkerStats\x12\".ray.rpc.GetCoreWorkerStatsRequest\x1a .ray.rpc.GetCoreWorkerStatsReply\x12\x39\n\x07LocalGC\x12\x17.ray.rpc.LocalGCRequest\x1a\x15.ray.rpc.LocalGCReply\x12K\n\rDeleteObjects\x12\x1d.ray.rpc.DeleteObjectsRequest\x1a\x1b.ray.rpc.DeleteObjectsReply\x12H\n\x0cSpillObjects\x12\x1c.ray.rpc.SpillObjectsRequest\x1a\x1a.ray.rpc.SpillObjectsReply\x12\x63\n\x15RestoreSpilledObjects\x12%.ray.rpc.RestoreSpilledObjectsRequest\x1a#.ray.rpc.RestoreSpilledObjectsReply\x12`\n\x14\x44\x65leteSpilledObjects\x12$.ray.rpc.DeleteSpilledObjectsRequest\x1a\".ray.rpc.DeleteSpilledObjectsReply\x12W\n\x11PlasmaObjectReady\x12!.ray.rpc.PlasmaObjectReadyRequest\x1a\x1f.ray.rpc.PlasmaObjectReadyReply\x12\x30\n\x04\x45xit\x12\x14.ray.rpc.ExitRequest\x1a\x12.ray.rpc.ExitReply\x12W\n\x11\x41ssignObjectOwner\x12!.ray.rpc.AssignObjectOwnerRequest\x1a\x1f.ray.rpc.AssignObjectOwnerReply\x12Q\n\x0fNumPendingTasks\x12\x1f.ray.rpc.NumPendingTasksRequest\x1a\x1d.ray.rpc.NumPendingTasksReply\x12u\n\x1bRegisterMutableObjectReader\x12+.ray.rpc.RegisterMutableObjectReaderRequest\x1a).ray.rpc.RegisterMutableObjectReaderReplyB\x03\xf8\x01\x01\x62\x06proto3')

_OBJECTPLASMALOCATIONUPDATE = DESCRIPTOR.enum_types_by_name['ObjectPlasmaLocationUpdate']
ObjectPlasmaLocationUpdate = enum_type_wrapper.EnumTypeWrapper(_OBJECTPLASMALOCATIONUPDATE)
ADDED = 0
REMOVED = 1


_ACTIVEOBJECTIDS = DESCRIPTOR.message_types_by_name['ActiveObjectIDs']
_ACTORHANDLE = DESCRIPTOR.message_types_by_name['ActorHandle']
_PUSHTASKREQUEST = DESCRIPTOR.message_types_by_name['PushTaskRequest']
_PUSHTASKREPLY = DESCRIPTOR.message_types_by_name['PushTaskReply']
_DIRECTACTORCALLARGWAITCOMPLETEREQUEST = DESCRIPTOR.message_types_by_name['DirectActorCallArgWaitCompleteRequest']
_DIRECTACTORCALLARGWAITCOMPLETEREPLY = DESCRIPTOR.message_types_by_name['DirectActorCallArgWaitCompleteReply']
_GETOBJECTSTATUSREQUEST = DESCRIPTOR.message_types_by_name['GetObjectStatusRequest']
_RAYOBJECT = DESCRIPTOR.message_types_by_name['RayObject']
_GETOBJECTSTATUSREPLY = DESCRIPTOR.message_types_by_name['GetObjectStatusReply']
_WAITFORACTORREFDELETEDREQUEST = DESCRIPTOR.message_types_by_name['WaitForActorRefDeletedRequest']
_WAITFORACTORREFDELETEDREPLY = DESCRIPTOR.message_types_by_name['WaitForActorRefDeletedReply']
_UPDATEOBJECTLOCATIONBATCHREQUEST = DESCRIPTOR.message_types_by_name['UpdateObjectLocationBatchRequest']
_UPDATEOBJECTLOCATIONBATCHREPLY = DESCRIPTOR.message_types_by_name['UpdateObjectLocationBatchReply']
_OBJECTSPILLEDLOCATIONUPDATE = DESCRIPTOR.message_types_by_name['ObjectSpilledLocationUpdate']
_OBJECTLOCATIONUPDATE = DESCRIPTOR.message_types_by_name['ObjectLocationUpdate']
_GETOBJECTLOCATIONSOWNERREQUEST = DESCRIPTOR.message_types_by_name['GetObjectLocationsOwnerRequest']
_GETOBJECTLOCATIONSOWNERREPLY = DESCRIPTOR.message_types_by_name['GetObjectLocationsOwnerReply']
_KILLACTORREQUEST = DESCRIPTOR.message_types_by_name['KillActorRequest']
_KILLACTORREPLY = DESCRIPTOR.message_types_by_name['KillActorReply']
_CANCELTASKREQUEST = DESCRIPTOR.message_types_by_name['CancelTaskRequest']
_CANCELTASKREPLY = DESCRIPTOR.message_types_by_name['CancelTaskReply']
_REMOTECANCELTASKREQUEST = DESCRIPTOR.message_types_by_name['RemoteCancelTaskRequest']
_REMOTECANCELTASKREPLY = DESCRIPTOR.message_types_by_name['RemoteCancelTaskReply']
_GETCOREWORKERSTATSREQUEST = DESCRIPTOR.message_types_by_name['GetCoreWorkerStatsRequest']
_GETCOREWORKERSTATSREPLY = DESCRIPTOR.message_types_by_name['GetCoreWorkerStatsReply']
_LOCALGCREQUEST = DESCRIPTOR.message_types_by_name['LocalGCRequest']
_LOCALGCREPLY = DESCRIPTOR.message_types_by_name['LocalGCReply']
_PLASMAOBJECTREADYREQUEST = DESCRIPTOR.message_types_by_name['PlasmaObjectReadyRequest']
_PLASMAOBJECTREADYREPLY = DESCRIPTOR.message_types_by_name['PlasmaObjectReadyReply']
_DELETEOBJECTSREQUEST = DESCRIPTOR.message_types_by_name['DeleteObjectsRequest']
_DELETEOBJECTSREPLY = DESCRIPTOR.message_types_by_name['DeleteObjectsReply']
_SPILLOBJECTSREQUEST = DESCRIPTOR.message_types_by_name['SpillObjectsRequest']
_SPILLOBJECTSREPLY = DESCRIPTOR.message_types_by_name['SpillObjectsReply']
_RESTORESPILLEDOBJECTSREQUEST = DESCRIPTOR.message_types_by_name['RestoreSpilledObjectsRequest']
_RESTORESPILLEDOBJECTSREPLY = DESCRIPTOR.message_types_by_name['RestoreSpilledObjectsReply']
_DELETESPILLEDOBJECTSREQUEST = DESCRIPTOR.message_types_by_name['DeleteSpilledObjectsRequest']
_DELETESPILLEDOBJECTSREPLY = DESCRIPTOR.message_types_by_name['DeleteSpilledObjectsReply']
_EXITREQUEST = DESCRIPTOR.message_types_by_name['ExitRequest']
_EXITREPLY = DESCRIPTOR.message_types_by_name['ExitReply']
_ASSIGNOBJECTOWNERREQUEST = DESCRIPTOR.message_types_by_name['AssignObjectOwnerRequest']
_ASSIGNOBJECTOWNERREPLY = DESCRIPTOR.message_types_by_name['AssignObjectOwnerReply']
_RAYLETNOTIFYGCSRESTARTREQUEST = DESCRIPTOR.message_types_by_name['RayletNotifyGCSRestartRequest']
_RAYLETNOTIFYGCSRESTARTREPLY = DESCRIPTOR.message_types_by_name['RayletNotifyGCSRestartReply']
_NUMPENDINGTASKSREQUEST = DESCRIPTOR.message_types_by_name['NumPendingTasksRequest']
_NUMPENDINGTASKSREPLY = DESCRIPTOR.message_types_by_name['NumPendingTasksReply']
_REPORTGENERATORITEMRETURNSREQUEST = DESCRIPTOR.message_types_by_name['ReportGeneratorItemReturnsRequest']
_REPORTGENERATORITEMRETURNSREPLY = DESCRIPTOR.message_types_by_name['ReportGeneratorItemReturnsReply']
_REGISTERMUTABLEOBJECTREADERREQUEST = DESCRIPTOR.message_types_by_name['RegisterMutableObjectReaderRequest']
_REGISTERMUTABLEOBJECTREADERREPLY = DESCRIPTOR.message_types_by_name['RegisterMutableObjectReaderReply']
_GETOBJECTSTATUSREPLY_OBJECTSTATUS = _GETOBJECTSTATUSREPLY.enum_types_by_name['ObjectStatus']
ActiveObjectIDs = _reflection.GeneratedProtocolMessageType('ActiveObjectIDs', (_message.Message,), {
  'DESCRIPTOR' : _ACTIVEOBJECTIDS,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ActiveObjectIDs)
  })
_sym_db.RegisterMessage(ActiveObjectIDs)

ActorHandle = _reflection.GeneratedProtocolMessageType('ActorHandle', (_message.Message,), {
  'DESCRIPTOR' : _ACTORHANDLE,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ActorHandle)
  })
_sym_db.RegisterMessage(ActorHandle)

PushTaskRequest = _reflection.GeneratedProtocolMessageType('PushTaskRequest', (_message.Message,), {
  'DESCRIPTOR' : _PUSHTASKREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.PushTaskRequest)
  })
_sym_db.RegisterMessage(PushTaskRequest)

PushTaskReply = _reflection.GeneratedProtocolMessageType('PushTaskReply', (_message.Message,), {
  'DESCRIPTOR' : _PUSHTASKREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.PushTaskReply)
  })
_sym_db.RegisterMessage(PushTaskReply)

DirectActorCallArgWaitCompleteRequest = _reflection.GeneratedProtocolMessageType('DirectActorCallArgWaitCompleteRequest', (_message.Message,), {
  'DESCRIPTOR' : _DIRECTACTORCALLARGWAITCOMPLETEREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.DirectActorCallArgWaitCompleteRequest)
  })
_sym_db.RegisterMessage(DirectActorCallArgWaitCompleteRequest)

DirectActorCallArgWaitCompleteReply = _reflection.GeneratedProtocolMessageType('DirectActorCallArgWaitCompleteReply', (_message.Message,), {
  'DESCRIPTOR' : _DIRECTACTORCALLARGWAITCOMPLETEREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.DirectActorCallArgWaitCompleteReply)
  })
_sym_db.RegisterMessage(DirectActorCallArgWaitCompleteReply)

GetObjectStatusRequest = _reflection.GeneratedProtocolMessageType('GetObjectStatusRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETOBJECTSTATUSREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.GetObjectStatusRequest)
  })
_sym_db.RegisterMessage(GetObjectStatusRequest)

RayObject = _reflection.GeneratedProtocolMessageType('RayObject', (_message.Message,), {
  'DESCRIPTOR' : _RAYOBJECT,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.RayObject)
  })
_sym_db.RegisterMessage(RayObject)

GetObjectStatusReply = _reflection.GeneratedProtocolMessageType('GetObjectStatusReply', (_message.Message,), {
  'DESCRIPTOR' : _GETOBJECTSTATUSREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.GetObjectStatusReply)
  })
_sym_db.RegisterMessage(GetObjectStatusReply)

WaitForActorRefDeletedRequest = _reflection.GeneratedProtocolMessageType('WaitForActorRefDeletedRequest', (_message.Message,), {
  'DESCRIPTOR' : _WAITFORACTORREFDELETEDREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.WaitForActorRefDeletedRequest)
  })
_sym_db.RegisterMessage(WaitForActorRefDeletedRequest)

WaitForActorRefDeletedReply = _reflection.GeneratedProtocolMessageType('WaitForActorRefDeletedReply', (_message.Message,), {
  'DESCRIPTOR' : _WAITFORACTORREFDELETEDREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.WaitForActorRefDeletedReply)
  })
_sym_db.RegisterMessage(WaitForActorRefDeletedReply)

UpdateObjectLocationBatchRequest = _reflection.GeneratedProtocolMessageType('UpdateObjectLocationBatchRequest', (_message.Message,), {
  'DESCRIPTOR' : _UPDATEOBJECTLOCATIONBATCHREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.UpdateObjectLocationBatchRequest)
  })
_sym_db.RegisterMessage(UpdateObjectLocationBatchRequest)

UpdateObjectLocationBatchReply = _reflection.GeneratedProtocolMessageType('UpdateObjectLocationBatchReply', (_message.Message,), {
  'DESCRIPTOR' : _UPDATEOBJECTLOCATIONBATCHREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.UpdateObjectLocationBatchReply)
  })
_sym_db.RegisterMessage(UpdateObjectLocationBatchReply)

ObjectSpilledLocationUpdate = _reflection.GeneratedProtocolMessageType('ObjectSpilledLocationUpdate', (_message.Message,), {
  'DESCRIPTOR' : _OBJECTSPILLEDLOCATIONUPDATE,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ObjectSpilledLocationUpdate)
  })
_sym_db.RegisterMessage(ObjectSpilledLocationUpdate)

ObjectLocationUpdate = _reflection.GeneratedProtocolMessageType('ObjectLocationUpdate', (_message.Message,), {
  'DESCRIPTOR' : _OBJECTLOCATIONUPDATE,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ObjectLocationUpdate)
  })
_sym_db.RegisterMessage(ObjectLocationUpdate)

GetObjectLocationsOwnerRequest = _reflection.GeneratedProtocolMessageType('GetObjectLocationsOwnerRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETOBJECTLOCATIONSOWNERREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.GetObjectLocationsOwnerRequest)
  })
_sym_db.RegisterMessage(GetObjectLocationsOwnerRequest)

GetObjectLocationsOwnerReply = _reflection.GeneratedProtocolMessageType('GetObjectLocationsOwnerReply', (_message.Message,), {
  'DESCRIPTOR' : _GETOBJECTLOCATIONSOWNERREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.GetObjectLocationsOwnerReply)
  })
_sym_db.RegisterMessage(GetObjectLocationsOwnerReply)

KillActorRequest = _reflection.GeneratedProtocolMessageType('KillActorRequest', (_message.Message,), {
  'DESCRIPTOR' : _KILLACTORREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.KillActorRequest)
  })
_sym_db.RegisterMessage(KillActorRequest)

KillActorReply = _reflection.GeneratedProtocolMessageType('KillActorReply', (_message.Message,), {
  'DESCRIPTOR' : _KILLACTORREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.KillActorReply)
  })
_sym_db.RegisterMessage(KillActorReply)

CancelTaskRequest = _reflection.GeneratedProtocolMessageType('CancelTaskRequest', (_message.Message,), {
  'DESCRIPTOR' : _CANCELTASKREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.CancelTaskRequest)
  })
_sym_db.RegisterMessage(CancelTaskRequest)

CancelTaskReply = _reflection.GeneratedProtocolMessageType('CancelTaskReply', (_message.Message,), {
  'DESCRIPTOR' : _CANCELTASKREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.CancelTaskReply)
  })
_sym_db.RegisterMessage(CancelTaskReply)

RemoteCancelTaskRequest = _reflection.GeneratedProtocolMessageType('RemoteCancelTaskRequest', (_message.Message,), {
  'DESCRIPTOR' : _REMOTECANCELTASKREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.RemoteCancelTaskRequest)
  })
_sym_db.RegisterMessage(RemoteCancelTaskRequest)

RemoteCancelTaskReply = _reflection.GeneratedProtocolMessageType('RemoteCancelTaskReply', (_message.Message,), {
  'DESCRIPTOR' : _REMOTECANCELTASKREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.RemoteCancelTaskReply)
  })
_sym_db.RegisterMessage(RemoteCancelTaskReply)

GetCoreWorkerStatsRequest = _reflection.GeneratedProtocolMessageType('GetCoreWorkerStatsRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETCOREWORKERSTATSREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.GetCoreWorkerStatsRequest)
  })
_sym_db.RegisterMessage(GetCoreWorkerStatsRequest)

GetCoreWorkerStatsReply = _reflection.GeneratedProtocolMessageType('GetCoreWorkerStatsReply', (_message.Message,), {
  'DESCRIPTOR' : _GETCOREWORKERSTATSREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.GetCoreWorkerStatsReply)
  })
_sym_db.RegisterMessage(GetCoreWorkerStatsReply)

LocalGCRequest = _reflection.GeneratedProtocolMessageType('LocalGCRequest', (_message.Message,), {
  'DESCRIPTOR' : _LOCALGCREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.LocalGCRequest)
  })
_sym_db.RegisterMessage(LocalGCRequest)

LocalGCReply = _reflection.GeneratedProtocolMessageType('LocalGCReply', (_message.Message,), {
  'DESCRIPTOR' : _LOCALGCREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.LocalGCReply)
  })
_sym_db.RegisterMessage(LocalGCReply)

PlasmaObjectReadyRequest = _reflection.GeneratedProtocolMessageType('PlasmaObjectReadyRequest', (_message.Message,), {
  'DESCRIPTOR' : _PLASMAOBJECTREADYREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.PlasmaObjectReadyRequest)
  })
_sym_db.RegisterMessage(PlasmaObjectReadyRequest)

PlasmaObjectReadyReply = _reflection.GeneratedProtocolMessageType('PlasmaObjectReadyReply', (_message.Message,), {
  'DESCRIPTOR' : _PLASMAOBJECTREADYREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.PlasmaObjectReadyReply)
  })
_sym_db.RegisterMessage(PlasmaObjectReadyReply)

DeleteObjectsRequest = _reflection.GeneratedProtocolMessageType('DeleteObjectsRequest', (_message.Message,), {
  'DESCRIPTOR' : _DELETEOBJECTSREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.DeleteObjectsRequest)
  })
_sym_db.RegisterMessage(DeleteObjectsRequest)

DeleteObjectsReply = _reflection.GeneratedProtocolMessageType('DeleteObjectsReply', (_message.Message,), {
  'DESCRIPTOR' : _DELETEOBJECTSREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.DeleteObjectsReply)
  })
_sym_db.RegisterMessage(DeleteObjectsReply)

SpillObjectsRequest = _reflection.GeneratedProtocolMessageType('SpillObjectsRequest', (_message.Message,), {
  'DESCRIPTOR' : _SPILLOBJECTSREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.SpillObjectsRequest)
  })
_sym_db.RegisterMessage(SpillObjectsRequest)

SpillObjectsReply = _reflection.GeneratedProtocolMessageType('SpillObjectsReply', (_message.Message,), {
  'DESCRIPTOR' : _SPILLOBJECTSREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.SpillObjectsReply)
  })
_sym_db.RegisterMessage(SpillObjectsReply)

RestoreSpilledObjectsRequest = _reflection.GeneratedProtocolMessageType('RestoreSpilledObjectsRequest', (_message.Message,), {
  'DESCRIPTOR' : _RESTORESPILLEDOBJECTSREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.RestoreSpilledObjectsRequest)
  })
_sym_db.RegisterMessage(RestoreSpilledObjectsRequest)

RestoreSpilledObjectsReply = _reflection.GeneratedProtocolMessageType('RestoreSpilledObjectsReply', (_message.Message,), {
  'DESCRIPTOR' : _RESTORESPILLEDOBJECTSREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.RestoreSpilledObjectsReply)
  })
_sym_db.RegisterMessage(RestoreSpilledObjectsReply)

DeleteSpilledObjectsRequest = _reflection.GeneratedProtocolMessageType('DeleteSpilledObjectsRequest', (_message.Message,), {
  'DESCRIPTOR' : _DELETESPILLEDOBJECTSREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.DeleteSpilledObjectsRequest)
  })
_sym_db.RegisterMessage(DeleteSpilledObjectsRequest)

DeleteSpilledObjectsReply = _reflection.GeneratedProtocolMessageType('DeleteSpilledObjectsReply', (_message.Message,), {
  'DESCRIPTOR' : _DELETESPILLEDOBJECTSREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.DeleteSpilledObjectsReply)
  })
_sym_db.RegisterMessage(DeleteSpilledObjectsReply)

ExitRequest = _reflection.GeneratedProtocolMessageType('ExitRequest', (_message.Message,), {
  'DESCRIPTOR' : _EXITREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ExitRequest)
  })
_sym_db.RegisterMessage(ExitRequest)

ExitReply = _reflection.GeneratedProtocolMessageType('ExitReply', (_message.Message,), {
  'DESCRIPTOR' : _EXITREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ExitReply)
  })
_sym_db.RegisterMessage(ExitReply)

AssignObjectOwnerRequest = _reflection.GeneratedProtocolMessageType('AssignObjectOwnerRequest', (_message.Message,), {
  'DESCRIPTOR' : _ASSIGNOBJECTOWNERREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.AssignObjectOwnerRequest)
  })
_sym_db.RegisterMessage(AssignObjectOwnerRequest)

AssignObjectOwnerReply = _reflection.GeneratedProtocolMessageType('AssignObjectOwnerReply', (_message.Message,), {
  'DESCRIPTOR' : _ASSIGNOBJECTOWNERREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.AssignObjectOwnerReply)
  })
_sym_db.RegisterMessage(AssignObjectOwnerReply)

RayletNotifyGCSRestartRequest = _reflection.GeneratedProtocolMessageType('RayletNotifyGCSRestartRequest', (_message.Message,), {
  'DESCRIPTOR' : _RAYLETNOTIFYGCSRESTARTREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.RayletNotifyGCSRestartRequest)
  })
_sym_db.RegisterMessage(RayletNotifyGCSRestartRequest)

RayletNotifyGCSRestartReply = _reflection.GeneratedProtocolMessageType('RayletNotifyGCSRestartReply', (_message.Message,), {
  'DESCRIPTOR' : _RAYLETNOTIFYGCSRESTARTREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.RayletNotifyGCSRestartReply)
  })
_sym_db.RegisterMessage(RayletNotifyGCSRestartReply)

NumPendingTasksRequest = _reflection.GeneratedProtocolMessageType('NumPendingTasksRequest', (_message.Message,), {
  'DESCRIPTOR' : _NUMPENDINGTASKSREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.NumPendingTasksRequest)
  })
_sym_db.RegisterMessage(NumPendingTasksRequest)

NumPendingTasksReply = _reflection.GeneratedProtocolMessageType('NumPendingTasksReply', (_message.Message,), {
  'DESCRIPTOR' : _NUMPENDINGTASKSREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.NumPendingTasksReply)
  })
_sym_db.RegisterMessage(NumPendingTasksReply)

ReportGeneratorItemReturnsRequest = _reflection.GeneratedProtocolMessageType('ReportGeneratorItemReturnsRequest', (_message.Message,), {
  'DESCRIPTOR' : _REPORTGENERATORITEMRETURNSREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ReportGeneratorItemReturnsRequest)
  })
_sym_db.RegisterMessage(ReportGeneratorItemReturnsRequest)

ReportGeneratorItemReturnsReply = _reflection.GeneratedProtocolMessageType('ReportGeneratorItemReturnsReply', (_message.Message,), {
  'DESCRIPTOR' : _REPORTGENERATORITEMRETURNSREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ReportGeneratorItemReturnsReply)
  })
_sym_db.RegisterMessage(ReportGeneratorItemReturnsReply)

RegisterMutableObjectReaderRequest = _reflection.GeneratedProtocolMessageType('RegisterMutableObjectReaderRequest', (_message.Message,), {
  'DESCRIPTOR' : _REGISTERMUTABLEOBJECTREADERREQUEST,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.RegisterMutableObjectReaderRequest)
  })
_sym_db.RegisterMessage(RegisterMutableObjectReaderRequest)

RegisterMutableObjectReaderReply = _reflection.GeneratedProtocolMessageType('RegisterMutableObjectReaderReply', (_message.Message,), {
  'DESCRIPTOR' : _REGISTERMUTABLEOBJECTREADERREPLY,
  '__module__' : 'src.ray.protobuf.core_worker_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.RegisterMutableObjectReaderReply)
  })
_sym_db.RegisterMessage(RegisterMutableObjectReaderReply)

_COREWORKERSERVICE = DESCRIPTOR.services_by_name['CoreWorkerService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\370\001\001'
  _OBJECTPLASMALOCATIONUPDATE._serialized_start=6416
  _OBJECTPLASMALOCATIONUPDATE._serialized_end=6468
  _ACTIVEOBJECTIDS._serialized_start=109
  _ACTIVEOBJECTIDS._serialized_end=157
  _ACTORHANDLE._serialized_start=160
  _ACTORHANDLE._serialized_end=807
  _PUSHTASKREQUEST._serialized_start=810
  _PUSHTASKREQUEST._serialized_end=1085
  _PUSHTASKREPLY._serialized_start=1088
  _PUSHTASKREPLY._serialized_end=1735
  _DIRECTACTORCALLARGWAITCOMPLETEREQUEST._serialized_start=1737
  _DIRECTACTORCALLARGWAITCOMPLETEREQUEST._serialized_end=1840
  _DIRECTACTORCALLARGWAITCOMPLETEREPLY._serialized_start=1842
  _DIRECTACTORCALLARGWAITCOMPLETEREPLY._serialized_end=1879
  _GETOBJECTSTATUSREQUEST._serialized_start=1881
  _GETOBJECTSTATUSREQUEST._serialized_end=1974
  _RAYOBJECT._serialized_start=1977
  _RAYOBJECT._serialized_end=2110
  _GETOBJECTSTATUSREPLY._serialized_start=2113
  _GETOBJECTSTATUSREPLY._serialized_end=2365
  _GETOBJECTSTATUSREPLY_OBJECTSTATUS._serialized_start=2309
  _GETOBJECTSTATUSREPLY_OBJECTSTATUS._serialized_end=2365
  _WAITFORACTORREFDELETEDREQUEST._serialized_start=2367
  _WAITFORACTORREFDELETEDREQUEST._serialized_end=2471
  _WAITFORACTORREFDELETEDREPLY._serialized_start=2473
  _WAITFORACTORREFDELETEDREPLY._serialized_end=2502
  _UPDATEOBJECTLOCATIONBATCHREQUEST._serialized_start=2505
  _UPDATEOBJECTLOCATIONBATCHREQUEST._serialized_end=2697
  _UPDATEOBJECTLOCATIONBATCHREPLY._serialized_start=2699
  _UPDATEOBJECTLOCATIONBATCHREPLY._serialized_end=2731
  _OBJECTSPILLEDLOCATIONUPDATE._serialized_start=2733
  _OBJECTSPILLEDLOCATIONUPDATE._serialized_end=2852
  _OBJECTLOCATIONUPDATE._serialized_start=2855
  _OBJECTLOCATIONUPDATE._serialized_end=3213
  _GETOBJECTLOCATIONSOWNERREQUEST._serialized_start=3215
  _GETOBJECTLOCATIONSOWNERREQUEST._serialized_end=3324
  _GETOBJECTLOCATIONSOWNERREPLY._serialized_start=3326
  _GETOBJECTLOCATIONSOWNERREPLY._serialized_end=3450
  _KILLACTORREQUEST._serialized_start=3453
  _KILLACTORREQUEST._serialized_end=3605
  _KILLACTORREPLY._serialized_start=3607
  _KILLACTORREPLY._serialized_end=3623
  _CANCELTASKREQUEST._serialized_start=3626
  _CANCELTASKREQUEST._serialized_end=3790
  _CANCELTASKREPLY._serialized_start=3792
  _CANCELTASKREPLY._serialized_end=3908
  _REMOTECANCELTASKREQUEST._serialized_start=3911
  _REMOTECANCELTASKREQUEST._serialized_end=4039
  _REMOTECANCELTASKREPLY._serialized_start=4041
  _REMOTECANCELTASKREPLY._serialized_end=4064
  _GETCOREWORKERSTATSREQUEST._serialized_start=4067
  _GETCOREWORKERSTATSREQUEST._serialized_end=4269
  _GETCOREWORKERSTATSREPLY._serialized_start=4272
  _GETCOREWORKERSTATSREPLY._serialized_end=4521
  _LOCALGCREQUEST._serialized_start=4523
  _LOCALGCREQUEST._serialized_end=4592
  _LOCALGCREPLY._serialized_start=4594
  _LOCALGCREPLY._serialized_end=4608
  _PLASMAOBJECTREADYREQUEST._serialized_start=4610
  _PLASMAOBJECTREADYREQUEST._serialized_end=4665
  _PLASMAOBJECTREADYREPLY._serialized_start=4667
  _PLASMAOBJECTREADYREPLY._serialized_end=4691
  _DELETEOBJECTSREQUEST._serialized_start=4693
  _DELETEOBJECTSREQUEST._serialized_end=4777
  _DELETEOBJECTSREPLY._serialized_start=4779
  _DELETEOBJECTSREPLY._serialized_end=4799
  _SPILLOBJECTSREQUEST._serialized_start=4802
  _SPILLOBJECTSREQUEST._serialized_end=4968
  _SPILLOBJECTSREPLY._serialized_start=4970
  _SPILLOBJECTSREPLY._serialized_end=5037
  _RESTORESPILLEDOBJECTSREQUEST._serialized_start=5040
  _RESTORESPILLEDOBJECTSREQUEST._serialized_end=5169
  _RESTORESPILLEDOBJECTSREPLY._serialized_start=5171
  _RESTORESPILLEDOBJECTSREPLY._serialized_end=5249
  _DELETESPILLEDOBJECTSREQUEST._serialized_start=5251
  _DELETESPILLEDOBJECTSREQUEST._serialized_end=5328
  _DELETESPILLEDOBJECTSREPLY._serialized_start=5330
  _DELETESPILLEDOBJECTSREPLY._serialized_end=5357
  _EXITREQUEST._serialized_start=5359
  _EXITREQUEST._serialized_end=5403
  _EXITREPLY._serialized_start=5405
  _EXITREPLY._serialized_end=5442
  _ASSIGNOBJECTOWNERREQUEST._serialized_start=5445
  _ASSIGNOBJECTOWNERREQUEST._serialized_end=5673
  _ASSIGNOBJECTOWNERREPLY._serialized_start=5675
  _ASSIGNOBJECTOWNERREPLY._serialized_end=5699
  _RAYLETNOTIFYGCSRESTARTREQUEST._serialized_start=5701
  _RAYLETNOTIFYGCSRESTARTREQUEST._serialized_end=5732
  _RAYLETNOTIFYGCSRESTARTREPLY._serialized_start=5734
  _RAYLETNOTIFYGCSRESTARTREPLY._serialized_end=5763
  _NUMPENDINGTASKSREQUEST._serialized_start=5765
  _NUMPENDINGTASKSREQUEST._serialized_end=5789
  _NUMPENDINGTASKSREPLY._serialized_start=5791
  _NUMPENDINGTASKSREPLY._serialized_end=5857
  _REPORTGENERATORITEMRETURNSREQUEST._serialized_start=5860
  _REPORTGENERATORITEMRETURNSREQUEST._serialized_end=6128
  _REPORTGENERATORITEMRETURNSREPLY._serialized_start=6130
  _REPORTGENERATORITEMRETURNSREPLY._serialized_end=6222
  _REGISTERMUTABLEOBJECTREADERREQUEST._serialized_start=6225
  _REGISTERMUTABLEOBJECTREADERREQUEST._serialized_end=6378
  _REGISTERMUTABLEOBJECTREADERREPLY._serialized_start=6380
  _REGISTERMUTABLEOBJECTREADERREPLY._serialized_end=6414
  _COREWORKERSERVICE._serialized_start=6471
  _COREWORKERSERVICE._serialized_end=8638
# @@protoc_insertion_point(module_scope)