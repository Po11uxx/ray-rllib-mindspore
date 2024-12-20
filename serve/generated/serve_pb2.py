# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/ray/protobuf/serve.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1csrc/ray/protobuf/serve.proto\x12\tray.serve\"\xcc\x06\n\x11\x41utoscalingConfig\x12!\n\x0cmin_replicas\x18\x01 \x01(\rR\x0bminReplicas\x12!\n\x0cmax_replicas\x18\x02 \x01(\rR\x0bmaxReplicas\x12,\n\x12metrics_interval_s\x18\x03 \x01(\x01R\x10metricsIntervalS\x12+\n\x12look_back_period_s\x18\x04 \x01(\x01R\x0flookBackPeriodS\x12)\n\x10smoothing_factor\x18\x05 \x01(\x01R\x0fsmoothingFactor\x12*\n\x11\x64ownscale_delay_s\x18\x06 \x01(\x01R\x0f\x64ownscaleDelayS\x12&\n\x0fupscale_delay_s\x18\x07 \x01(\x01R\rupscaleDelayS\x12.\n\x10initial_replicas\x18\x08 \x01(\rH\x00R\x0finitialReplicas\x88\x01\x01\x12=\n\x18upscale_smoothing_factor\x18\t \x01(\x01H\x01R\x16upscaleSmoothingFactor\x88\x01\x01\x12\x41\n\x1a\x64ownscale_smoothing_factor\x18\n \x01(\x01H\x02R\x18\x64ownscaleSmoothingFactor\x88\x01\x01\x12\x33\n\x16_serialized_policy_def\x18\x0b \x01(\x0cR\x13SerializedPolicyDef\x12\x17\n\x07_policy\x18\x0c \x01(\tR\x06Policy\x12\x36\n\x17target_ongoing_requests\x18\r \x01(\x01R\x15targetOngoingRequests\x12.\n\x10upscaling_factor\x18\x0e \x01(\x01H\x03R\x0fupscalingFactor\x88\x01\x01\x12\x32\n\x12\x64ownscaling_factor\x18\x0f \x01(\x01H\x04R\x11\x64ownscalingFactor\x88\x01\x01\x42\x13\n\x11_initial_replicasB\x1b\n\x19_upscale_smoothing_factorB\x1d\n\x1b_downscale_smoothing_factorB\x13\n\x11_upscaling_factorB\x15\n\x13_downscaling_factor\"\xa8\x01\n\rLoggingConfig\x12\x33\n\x08\x65ncoding\x18\x01 \x01(\x0e\x32\x17.ray.serve.EncodingTypeR\x08\x65ncoding\x12\x1b\n\tlog_level\x18\x02 \x01(\tR\x08logLevel\x12\x19\n\x08logs_dir\x18\x03 \x01(\tR\x07logsDir\x12*\n\x11\x65nable_access_log\x18\x04 \x01(\x08R\x0f\x65nableAccessLog\"\x86\x06\n\x10\x44\x65ploymentConfig\x12!\n\x0cnum_replicas\x18\x01 \x01(\x05R\x0bnumReplicas\x12\x30\n\x14max_ongoing_requests\x18\x02 \x01(\x05R\x12maxOngoingRequests\x12.\n\x13max_queued_requests\x18\x03 \x01(\x05R\x11maxQueuedRequests\x12\x1f\n\x0buser_config\x18\x04 \x01(\x0cR\nuserConfig\x12@\n\x1dgraceful_shutdown_wait_loop_s\x18\x05 \x01(\x01R\x19gracefulShutdownWaitLoopS\x12=\n\x1bgraceful_shutdown_timeout_s\x18\x06 \x01(\x01R\x18gracefulShutdownTimeoutS\x12\x31\n\x15health_check_period_s\x18\x07 \x01(\x01R\x12healthCheckPeriodS\x12\x33\n\x16health_check_timeout_s\x18\x08 \x01(\x01R\x13healthCheckTimeoutS\x12*\n\x11is_cross_language\x18\t \x01(\x08R\x0fisCrossLanguage\x12N\n\x13\x64\x65ployment_language\x18\n \x01(\x0e\x32\x1d.ray.serve.DeploymentLanguageR\x12\x64\x65ploymentLanguage\x12K\n\x12\x61utoscaling_config\x18\x0b \x01(\x0b\x32\x1c.ray.serve.AutoscalingConfigR\x11\x61utoscalingConfig\x12\x18\n\x07version\x18\x0c \x01(\tR\x07version\x12?\n\x1cuser_configured_option_names\x18\r \x03(\tR\x19userConfiguredOptionNames\x12?\n\x0elogging_config\x18\x0e \x01(\x0b\x32\x18.ray.serve.LoggingConfigR\rloggingConfig\"\xe4\x02\n\x0fRequestMetadata\x12\x1d\n\nrequest_id\x18\x01 \x01(\tR\trequestId\x12\x1a\n\x08\x65ndpoint\x18\x02 \x01(\tR\x08\x65ndpoint\x12\x1f\n\x0b\x63\x61ll_method\x18\x03 \x01(\tR\ncallMethod\x12\x41\n\x07\x63ontext\x18\x04 \x03(\x0b\x32\'.ray.serve.RequestMetadata.ContextEntryR\x07\x63ontext\x12\x30\n\x14multiplexed_model_id\x18\x05 \x01(\tR\x12multiplexedModelId\x12\x14\n\x05route\x18\x06 \x01(\tR\x05route\x12.\n\x13internal_request_id\x18\x07 \x01(\tR\x11internalRequestId\x1a:\n\x0c\x43ontextEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value:\x02\x38\x01\"$\n\x0eRequestWrapper\x12\x12\n\x04\x62ody\x18\x01 \x01(\x0cR\x04\x62ody\"Y\n\rUpdatedObject\x12\'\n\x0fobject_snapshot\x18\x01 \x01(\x0cR\x0eobjectSnapshot\x12\x1f\n\x0bsnapshot_id\x18\x02 \x01(\x05R\nsnapshotId\"\xbb\x01\n\x0fLongPollRequest\x12\x62\n\x14keys_to_snapshot_ids\x18\x01 \x03(\x0b\x32\x31.ray.serve.LongPollRequest.KeysToSnapshotIdsEntryR\x11keysToSnapshotIds\x1a\x44\n\x16KeysToSnapshotIdsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\x05R\x05value:\x02\x38\x01\"\xc5\x01\n\x0eLongPollResult\x12V\n\x0fupdated_objects\x18\x01 \x03(\x0b\x32-.ray.serve.LongPollResult.UpdatedObjectsEntryR\x0eupdatedObjects\x1a[\n\x13UpdatedObjectsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12.\n\x05value\x18\x02 \x01(\x0b\x32\x18.ray.serve.UpdatedObjectR\x05value:\x02\x38\x01\"\xc1\x01\n\x0c\x45ndpointInfo\x12#\n\rendpoint_name\x18\x01 \x01(\tR\x0c\x65ndpointName\x12\x14\n\x05route\x18\x02 \x01(\tR\x05route\x12;\n\x06\x63onfig\x18\x03 \x03(\x0b\x32#.ray.serve.EndpointInfo.ConfigEntryR\x06\x63onfig\x1a\x39\n\x0b\x43onfigEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value:\x02\x38\x01\"\xa9\x01\n\x0b\x45ndpointSet\x12\x43\n\tendpoints\x18\x01 \x03(\x0b\x32%.ray.serve.EndpointSet.EndpointsEntryR\tendpoints\x1aU\n\x0e\x45ndpointsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12-\n\x05value\x18\x02 \x01(\x0b\x32\x17.ray.serve.EndpointInfoR\x05value:\x02\x38\x01\"%\n\rActorNameList\x12\x14\n\x05names\x18\x01 \x03(\tR\x05names\"\xd1\x02\n\x11\x44\x65ploymentVersion\x12!\n\x0c\x63ode_version\x18\x01 \x01(\tR\x0b\x63odeVersion\x12H\n\x11\x64\x65ployment_config\x18\x02 \x01(\x0b\x32\x1b.ray.serve.DeploymentConfigR\x10\x64\x65ploymentConfig\x12*\n\x11ray_actor_options\x18\x03 \x01(\tR\x0frayActorOptions\x12\x36\n\x17placement_group_bundles\x18\x04 \x01(\tR\x15placementGroupBundles\x12\x38\n\x18placement_group_strategy\x18\x05 \x01(\tR\x16placementGroupStrategy\x12\x31\n\x15max_replicas_per_node\x18\x06 \x01(\x05R\x12maxReplicasPerNode\"\xf5\x02\n\rReplicaConfig\x12.\n\x13\x64\x65ployment_def_name\x18\x01 \x01(\tR\x11\x64\x65ploymentDefName\x12%\n\x0e\x64\x65ployment_def\x18\x02 \x01(\x0cR\rdeploymentDef\x12\x1b\n\tinit_args\x18\x03 \x01(\x0cR\x08initArgs\x12\x1f\n\x0binit_kwargs\x18\x04 \x01(\x0cR\ninitKwargs\x12*\n\x11ray_actor_options\x18\x05 \x01(\tR\x0frayActorOptions\x12\x36\n\x17placement_group_bundles\x18\x06 \x01(\tR\x15placementGroupBundles\x12\x38\n\x18placement_group_strategy\x18\x07 \x01(\tR\x16placementGroupStrategy\x12\x31\n\x15max_replicas_per_node\x18\x08 \x01(\x05R\x12maxReplicasPerNode\"\xb5\x03\n\x0e\x44\x65ploymentInfo\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12H\n\x11\x64\x65ployment_config\x18\x02 \x01(\x0b\x32\x1b.ray.serve.DeploymentConfigR\x10\x64\x65ploymentConfig\x12?\n\x0ereplica_config\x18\x03 \x01(\x0b\x32\x18.ray.serve.ReplicaConfigR\rreplicaConfig\x12\"\n\rstart_time_ms\x18\x04 \x01(\x03R\x0bstartTimeMs\x12\x1d\n\nactor_name\x18\x05 \x01(\tR\tactorName\x12\x18\n\x07version\x18\x06 \x01(\tR\x07version\x12\x1e\n\x0b\x65nd_time_ms\x18\x07 \x01(\x03R\tendTimeMs\x12\'\n\x0ftarget_capacity\x18\x08 \x01(\x01R\x0etargetCapacity\x12^\n\x19target_capacity_direction\x18\t \x01(\x0e\x32\".ray.serve.TargetCapacityDirectionR\x17targetCapacityDirection\"k\n\x0f\x44\x65ploymentRoute\x12\x42\n\x0f\x64\x65ployment_info\x18\x01 \x01(\x0b\x32\x19.ray.serve.DeploymentInfoR\x0e\x64\x65ploymentInfo\x12\x14\n\x05route\x18\x02 \x01(\tR\x05route\"^\n\x13\x44\x65ploymentRouteList\x12G\n\x11\x64\x65ployment_routes\x18\x01 \x03(\x0b\x32\x1a.ray.serve.DeploymentRouteR\x10\x64\x65ploymentRoutes\"\xc4\x01\n\x14\x44\x65ploymentStatusInfo\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x33\n\x06status\x18\x02 \x01(\x0e\x32\x1b.ray.serve.DeploymentStatusR\x06status\x12\x18\n\x07message\x18\x03 \x01(\tR\x07message\x12I\n\x0estatus_trigger\x18\x04 \x01(\x0e\x32\".ray.serve.DeploymentStatusTriggerR\rstatusTrigger\"s\n\x18\x44\x65ploymentStatusInfoList\x12W\n\x17\x64\x65ployment_status_infos\x18\x01 \x03(\x0b\x32\x1f.ray.serve.DeploymentStatusInfoR\x15\x64\x65ploymentStatusInfos\"\x9a\x01\n\x15\x41pplicationStatusInfo\x12\x34\n\x06status\x18\x01 \x01(\x0e\x32\x1c.ray.serve.ApplicationStatusR\x06status\x12\x18\n\x07message\x18\x02 \x01(\tR\x07message\x12\x31\n\x14\x64\x65ployment_timestamp\x18\x03 \x01(\x01R\x13\x64\x65ploymentTimestamp\"\xbb\x01\n\x0eStatusOverview\x12?\n\napp_status\x18\x01 \x01(\x0b\x32 .ray.serve.ApplicationStatusInfoR\tappStatus\x12T\n\x13\x64\x65ployment_statuses\x18\x02 \x01(\x0b\x32#.ray.serve.DeploymentStatusInfoListR\x12\x64\x65ploymentStatuses\x12\x12\n\x04name\x18\x03 \x01(\tR\x04name\"\x19\n\x17ListApplicationsRequest\"G\n\x18ListApplicationsResponse\x12+\n\x11\x61pplication_names\x18\x01 \x03(\tR\x10\x61pplicationNames\"\x10\n\x0eHealthzRequest\"+\n\x0fHealthzResponse\x12\x18\n\x07message\x18\x01 \x01(\tR\x07message\"L\n\x12UserDefinedMessage\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x10\n\x03\x66oo\x18\x02 \x01(\tR\x03\x66oo\x12\x10\n\x03num\x18\x03 \x01(\x03R\x03num\"H\n\x13UserDefinedResponse\x12\x1a\n\x08greeting\x18\x01 \x01(\tR\x08greeting\x12\x15\n\x06num_x2\x18\x02 \x01(\x03R\x05numX2\"\x15\n\x13UserDefinedMessage2\"2\n\x14UserDefinedResponse2\x12\x1a\n\x08greeting\x18\x01 \x01(\tR\x08greeting\"T\n\x0c\x46ruitAmounts\x12\x16\n\x06orange\x18\x01 \x01(\x03R\x06orange\x12\x14\n\x05\x61pple\x18\x02 \x01(\x03R\x05\x61pple\x12\x16\n\x06\x62\x61nana\x18\x03 \x01(\x03R\x06\x62\x61nana\"\"\n\nFruitCosts\x12\x14\n\x05\x63osts\x18\x01 \x01(\x02R\x05\x63osts\"\x1f\n\tArrayData\x12\x12\n\x04nums\x18\x01 \x03(\x02R\x04nums\" \n\nStringData\x12\x12\n\x04\x64\x61ta\x18\x01 \x01(\tR\x04\x64\x61ta\"%\n\x0bModelOutput\x12\x16\n\x06output\x18\x01 \x01(\x02R\x06output\"\xb8\x02\n\x0e\x44\x65ploymentArgs\x12\'\n\x0f\x64\x65ployment_name\x18\x01 \x01(\tR\x0e\x64\x65ploymentName\x12+\n\x11\x64\x65ployment_config\x18\x02 \x01(\x0cR\x10\x64\x65ploymentConfig\x12%\n\x0ereplica_config\x18\x03 \x01(\x0cR\rreplicaConfig\x12&\n\x0f\x64\x65ployer_job_id\x18\x04 \x01(\tR\rdeployerJobId\x12&\n\x0croute_prefix\x18\x05 \x01(\tH\x00R\x0broutePrefix\x88\x01\x01\x12\x18\n\x07ingress\x18\x06 \x01(\x08R\x07ingress\x12 \n\tdocs_path\x18\x07 \x01(\tH\x01R\x08\x64ocsPath\x88\x01\x01\x42\x0f\n\r_route_prefixB\x0c\n\n_docs_path*\"\n\x0c\x45ncodingType\x12\x08\n\x04TEXT\x10\x00\x12\x08\n\x04JSON\x10\x01**\n\x12\x44\x65ploymentLanguage\x12\n\n\x06PYTHON\x10\x00\x12\x08\n\x04JAVA\x10\x01*6\n\x17TargetCapacityDirection\x12\t\n\x05UNSET\x10\x00\x12\x06\n\x02UP\x10\x01\x12\x08\n\x04\x44OWN\x10\x02*\xb6\x01\n\x10\x44\x65ploymentStatus\x12\x1e\n\x1a\x44\x45PLOYMENT_STATUS_UPDATING\x10\x00\x12\x1d\n\x19\x44\x45PLOYMENT_STATUS_HEALTHY\x10\x01\x12\x1f\n\x1b\x44\x45PLOYMENT_STATUS_UNHEALTHY\x10\x02\x12\x1f\n\x1b\x44\x45PLOYMENT_STATUS_UPSCALING\x10\x03\x12!\n\x1d\x44\x45PLOYMENT_STATUS_DOWNSCALING\x10\x04*\xfe\x03\n\x17\x44\x65ploymentStatusTrigger\x12)\n%DEPLOYMENT_STATUS_TRIGGER_UNSPECIFIED\x10\x00\x12\x33\n/DEPLOYMENT_STATUS_TRIGGER_CONFIG_UPDATE_STARTED\x10\x01\x12\x35\n1DEPLOYMENT_STATUS_TRIGGER_CONFIG_UPDATE_COMPLETED\x10\x02\x12/\n+DEPLOYMENT_STATUS_TRIGGER_UPSCALE_COMPLETED\x10\x03\x12\x31\n-DEPLOYMENT_STATUS_TRIGGER_DOWNSCALE_COMPLETED\x10\x04\x12)\n%DEPLOYMENT_STATUS_TRIGGER_AUTOSCALING\x10\x05\x12\x34\n0DEPLOYMENT_STATUS_TRIGGER_REPLICA_STARTUP_FAILED\x10\x06\x12\x31\n-DEPLOYMENT_STATUS_TRIGGER_HEALTH_CHECK_FAILED\x10\x07\x12,\n(DEPLOYMENT_STATUS_TRIGGER_INTERNAL_ERROR\x10\x08\x12&\n\"DEPLOYMENT_STATUS_TRIGGER_DELETING\x10\t*\xe2\x01\n\x11\x41pplicationStatus\x12 \n\x1c\x41PPLICATION_STATUS_DEPLOYING\x10\x00\x12\x1e\n\x1a\x41PPLICATION_STATUS_RUNNING\x10\x01\x12$\n APPLICATION_STATUS_DEPLOY_FAILED\x10\x02\x12\x1f\n\x1b\x41PPLICATION_STATUS_DELETING\x10\x03\x12\"\n\x1e\x41PPLICATION_STATUS_NOT_STARTED\x10\x05\x12 \n\x1c\x41PPLICATION_STATUS_UNHEALTHY\x10\x06\x32\xb3\x01\n\x12RayServeAPIService\x12[\n\x10ListApplications\x12\".ray.serve.ListApplicationsRequest\x1a#.ray.serve.ListApplicationsResponse\x12@\n\x07Healthz\x12\x19.ray.serve.HealthzRequest\x1a\x1a.ray.serve.HealthzResponse2\xc3\x02\n\x12UserDefinedService\x12I\n\x08__call__\x12\x1d.ray.serve.UserDefinedMessage\x1a\x1e.ray.serve.UserDefinedResponse\x12H\n\x07Method1\x12\x1d.ray.serve.UserDefinedMessage\x1a\x1e.ray.serve.UserDefinedResponse\x12J\n\x07Method2\x12\x1e.ray.serve.UserDefinedMessage2\x1a\x1f.ray.serve.UserDefinedResponse2\x12L\n\tStreaming\x12\x1d.ray.serve.UserDefinedMessage\x1a\x1e.ray.serve.UserDefinedResponse0\x01\x32L\n\x0c\x46ruitService\x12<\n\nFruitStand\x12\x17.ray.serve.FruitAmounts\x1a\x15.ray.serve.FruitCosts2\x98\x01\n\x18RayServeBenchmarkService\x12\x39\n\tgrpc_call\x12\x14.ray.serve.ArrayData\x1a\x16.ray.serve.ModelOutput\x12\x41\n\x10\x63\x61ll_with_string\x12\x15.ray.serve.StringData\x1a\x16.ray.serve.ModelOutputB*\n\x16io.ray.serve.generatedB\x0bServeProtosP\x01\xf8\x01\x01\x62\x06proto3')

_ENCODINGTYPE = DESCRIPTOR.enum_types_by_name['EncodingType']
EncodingType = enum_type_wrapper.EnumTypeWrapper(_ENCODINGTYPE)
_DEPLOYMENTLANGUAGE = DESCRIPTOR.enum_types_by_name['DeploymentLanguage']
DeploymentLanguage = enum_type_wrapper.EnumTypeWrapper(_DEPLOYMENTLANGUAGE)
_TARGETCAPACITYDIRECTION = DESCRIPTOR.enum_types_by_name['TargetCapacityDirection']
TargetCapacityDirection = enum_type_wrapper.EnumTypeWrapper(_TARGETCAPACITYDIRECTION)
_DEPLOYMENTSTATUS = DESCRIPTOR.enum_types_by_name['DeploymentStatus']
DeploymentStatus = enum_type_wrapper.EnumTypeWrapper(_DEPLOYMENTSTATUS)
_DEPLOYMENTSTATUSTRIGGER = DESCRIPTOR.enum_types_by_name['DeploymentStatusTrigger']
DeploymentStatusTrigger = enum_type_wrapper.EnumTypeWrapper(_DEPLOYMENTSTATUSTRIGGER)
_APPLICATIONSTATUS = DESCRIPTOR.enum_types_by_name['ApplicationStatus']
ApplicationStatus = enum_type_wrapper.EnumTypeWrapper(_APPLICATIONSTATUS)
TEXT = 0
JSON = 1
PYTHON = 0
JAVA = 1
UNSET = 0
UP = 1
DOWN = 2
DEPLOYMENT_STATUS_UPDATING = 0
DEPLOYMENT_STATUS_HEALTHY = 1
DEPLOYMENT_STATUS_UNHEALTHY = 2
DEPLOYMENT_STATUS_UPSCALING = 3
DEPLOYMENT_STATUS_DOWNSCALING = 4
DEPLOYMENT_STATUS_TRIGGER_UNSPECIFIED = 0
DEPLOYMENT_STATUS_TRIGGER_CONFIG_UPDATE_STARTED = 1
DEPLOYMENT_STATUS_TRIGGER_CONFIG_UPDATE_COMPLETED = 2
DEPLOYMENT_STATUS_TRIGGER_UPSCALE_COMPLETED = 3
DEPLOYMENT_STATUS_TRIGGER_DOWNSCALE_COMPLETED = 4
DEPLOYMENT_STATUS_TRIGGER_AUTOSCALING = 5
DEPLOYMENT_STATUS_TRIGGER_REPLICA_STARTUP_FAILED = 6
DEPLOYMENT_STATUS_TRIGGER_HEALTH_CHECK_FAILED = 7
DEPLOYMENT_STATUS_TRIGGER_INTERNAL_ERROR = 8
DEPLOYMENT_STATUS_TRIGGER_DELETING = 9
APPLICATION_STATUS_DEPLOYING = 0
APPLICATION_STATUS_RUNNING = 1
APPLICATION_STATUS_DEPLOY_FAILED = 2
APPLICATION_STATUS_DELETING = 3
APPLICATION_STATUS_NOT_STARTED = 5
APPLICATION_STATUS_UNHEALTHY = 6


_AUTOSCALINGCONFIG = DESCRIPTOR.message_types_by_name['AutoscalingConfig']
_LOGGINGCONFIG = DESCRIPTOR.message_types_by_name['LoggingConfig']
_DEPLOYMENTCONFIG = DESCRIPTOR.message_types_by_name['DeploymentConfig']
_REQUESTMETADATA = DESCRIPTOR.message_types_by_name['RequestMetadata']
_REQUESTMETADATA_CONTEXTENTRY = _REQUESTMETADATA.nested_types_by_name['ContextEntry']
_REQUESTWRAPPER = DESCRIPTOR.message_types_by_name['RequestWrapper']
_UPDATEDOBJECT = DESCRIPTOR.message_types_by_name['UpdatedObject']
_LONGPOLLREQUEST = DESCRIPTOR.message_types_by_name['LongPollRequest']
_LONGPOLLREQUEST_KEYSTOSNAPSHOTIDSENTRY = _LONGPOLLREQUEST.nested_types_by_name['KeysToSnapshotIdsEntry']
_LONGPOLLRESULT = DESCRIPTOR.message_types_by_name['LongPollResult']
_LONGPOLLRESULT_UPDATEDOBJECTSENTRY = _LONGPOLLRESULT.nested_types_by_name['UpdatedObjectsEntry']
_ENDPOINTINFO = DESCRIPTOR.message_types_by_name['EndpointInfo']
_ENDPOINTINFO_CONFIGENTRY = _ENDPOINTINFO.nested_types_by_name['ConfigEntry']
_ENDPOINTSET = DESCRIPTOR.message_types_by_name['EndpointSet']
_ENDPOINTSET_ENDPOINTSENTRY = _ENDPOINTSET.nested_types_by_name['EndpointsEntry']
_ACTORNAMELIST = DESCRIPTOR.message_types_by_name['ActorNameList']
_DEPLOYMENTVERSION = DESCRIPTOR.message_types_by_name['DeploymentVersion']
_REPLICACONFIG = DESCRIPTOR.message_types_by_name['ReplicaConfig']
_DEPLOYMENTINFO = DESCRIPTOR.message_types_by_name['DeploymentInfo']
_DEPLOYMENTROUTE = DESCRIPTOR.message_types_by_name['DeploymentRoute']
_DEPLOYMENTROUTELIST = DESCRIPTOR.message_types_by_name['DeploymentRouteList']
_DEPLOYMENTSTATUSINFO = DESCRIPTOR.message_types_by_name['DeploymentStatusInfo']
_DEPLOYMENTSTATUSINFOLIST = DESCRIPTOR.message_types_by_name['DeploymentStatusInfoList']
_APPLICATIONSTATUSINFO = DESCRIPTOR.message_types_by_name['ApplicationStatusInfo']
_STATUSOVERVIEW = DESCRIPTOR.message_types_by_name['StatusOverview']
_LISTAPPLICATIONSREQUEST = DESCRIPTOR.message_types_by_name['ListApplicationsRequest']
_LISTAPPLICATIONSRESPONSE = DESCRIPTOR.message_types_by_name['ListApplicationsResponse']
_HEALTHZREQUEST = DESCRIPTOR.message_types_by_name['HealthzRequest']
_HEALTHZRESPONSE = DESCRIPTOR.message_types_by_name['HealthzResponse']
_USERDEFINEDMESSAGE = DESCRIPTOR.message_types_by_name['UserDefinedMessage']
_USERDEFINEDRESPONSE = DESCRIPTOR.message_types_by_name['UserDefinedResponse']
_USERDEFINEDMESSAGE2 = DESCRIPTOR.message_types_by_name['UserDefinedMessage2']
_USERDEFINEDRESPONSE2 = DESCRIPTOR.message_types_by_name['UserDefinedResponse2']
_FRUITAMOUNTS = DESCRIPTOR.message_types_by_name['FruitAmounts']
_FRUITCOSTS = DESCRIPTOR.message_types_by_name['FruitCosts']
_ARRAYDATA = DESCRIPTOR.message_types_by_name['ArrayData']
_STRINGDATA = DESCRIPTOR.message_types_by_name['StringData']
_MODELOUTPUT = DESCRIPTOR.message_types_by_name['ModelOutput']
_DEPLOYMENTARGS = DESCRIPTOR.message_types_by_name['DeploymentArgs']
AutoscalingConfig = _reflection.GeneratedProtocolMessageType('AutoscalingConfig', (_message.Message,), {
  'DESCRIPTOR' : _AUTOSCALINGCONFIG,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.AutoscalingConfig)
  })
_sym_db.RegisterMessage(AutoscalingConfig)

LoggingConfig = _reflection.GeneratedProtocolMessageType('LoggingConfig', (_message.Message,), {
  'DESCRIPTOR' : _LOGGINGCONFIG,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.LoggingConfig)
  })
_sym_db.RegisterMessage(LoggingConfig)

DeploymentConfig = _reflection.GeneratedProtocolMessageType('DeploymentConfig', (_message.Message,), {
  'DESCRIPTOR' : _DEPLOYMENTCONFIG,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.DeploymentConfig)
  })
_sym_db.RegisterMessage(DeploymentConfig)

RequestMetadata = _reflection.GeneratedProtocolMessageType('RequestMetadata', (_message.Message,), {

  'ContextEntry' : _reflection.GeneratedProtocolMessageType('ContextEntry', (_message.Message,), {
    'DESCRIPTOR' : _REQUESTMETADATA_CONTEXTENTRY,
    '__module__' : 'ray.serve.generated.serve_pb2'
    # @@protoc_insertion_point(class_scope:ray.serve.RequestMetadata.ContextEntry)
    })
  ,
  'DESCRIPTOR' : _REQUESTMETADATA,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.RequestMetadata)
  })
_sym_db.RegisterMessage(RequestMetadata)
_sym_db.RegisterMessage(RequestMetadata.ContextEntry)

RequestWrapper = _reflection.GeneratedProtocolMessageType('RequestWrapper', (_message.Message,), {
  'DESCRIPTOR' : _REQUESTWRAPPER,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.RequestWrapper)
  })
_sym_db.RegisterMessage(RequestWrapper)

UpdatedObject = _reflection.GeneratedProtocolMessageType('UpdatedObject', (_message.Message,), {
  'DESCRIPTOR' : _UPDATEDOBJECT,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.UpdatedObject)
  })
_sym_db.RegisterMessage(UpdatedObject)

LongPollRequest = _reflection.GeneratedProtocolMessageType('LongPollRequest', (_message.Message,), {

  'KeysToSnapshotIdsEntry' : _reflection.GeneratedProtocolMessageType('KeysToSnapshotIdsEntry', (_message.Message,), {
    'DESCRIPTOR' : _LONGPOLLREQUEST_KEYSTOSNAPSHOTIDSENTRY,
    '__module__' : 'ray.serve.generated.serve_pb2'
    # @@protoc_insertion_point(class_scope:ray.serve.LongPollRequest.KeysToSnapshotIdsEntry)
    })
  ,
  'DESCRIPTOR' : _LONGPOLLREQUEST,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.LongPollRequest)
  })
_sym_db.RegisterMessage(LongPollRequest)
_sym_db.RegisterMessage(LongPollRequest.KeysToSnapshotIdsEntry)

LongPollResult = _reflection.GeneratedProtocolMessageType('LongPollResult', (_message.Message,), {

  'UpdatedObjectsEntry' : _reflection.GeneratedProtocolMessageType('UpdatedObjectsEntry', (_message.Message,), {
    'DESCRIPTOR' : _LONGPOLLRESULT_UPDATEDOBJECTSENTRY,
    '__module__' : 'ray.serve.generated.serve_pb2'
    # @@protoc_insertion_point(class_scope:ray.serve.LongPollResult.UpdatedObjectsEntry)
    })
  ,
  'DESCRIPTOR' : _LONGPOLLRESULT,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.LongPollResult)
  })
_sym_db.RegisterMessage(LongPollResult)
_sym_db.RegisterMessage(LongPollResult.UpdatedObjectsEntry)

EndpointInfo = _reflection.GeneratedProtocolMessageType('EndpointInfo', (_message.Message,), {

  'ConfigEntry' : _reflection.GeneratedProtocolMessageType('ConfigEntry', (_message.Message,), {
    'DESCRIPTOR' : _ENDPOINTINFO_CONFIGENTRY,
    '__module__' : 'ray.serve.generated.serve_pb2'
    # @@protoc_insertion_point(class_scope:ray.serve.EndpointInfo.ConfigEntry)
    })
  ,
  'DESCRIPTOR' : _ENDPOINTINFO,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.EndpointInfo)
  })
_sym_db.RegisterMessage(EndpointInfo)
_sym_db.RegisterMessage(EndpointInfo.ConfigEntry)

EndpointSet = _reflection.GeneratedProtocolMessageType('EndpointSet', (_message.Message,), {

  'EndpointsEntry' : _reflection.GeneratedProtocolMessageType('EndpointsEntry', (_message.Message,), {
    'DESCRIPTOR' : _ENDPOINTSET_ENDPOINTSENTRY,
    '__module__' : 'ray.serve.generated.serve_pb2'
    # @@protoc_insertion_point(class_scope:ray.serve.EndpointSet.EndpointsEntry)
    })
  ,
  'DESCRIPTOR' : _ENDPOINTSET,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.EndpointSet)
  })
_sym_db.RegisterMessage(EndpointSet)
_sym_db.RegisterMessage(EndpointSet.EndpointsEntry)

ActorNameList = _reflection.GeneratedProtocolMessageType('ActorNameList', (_message.Message,), {
  'DESCRIPTOR' : _ACTORNAMELIST,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.ActorNameList)
  })
_sym_db.RegisterMessage(ActorNameList)

DeploymentVersion = _reflection.GeneratedProtocolMessageType('DeploymentVersion', (_message.Message,), {
  'DESCRIPTOR' : _DEPLOYMENTVERSION,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.DeploymentVersion)
  })
_sym_db.RegisterMessage(DeploymentVersion)

ReplicaConfig = _reflection.GeneratedProtocolMessageType('ReplicaConfig', (_message.Message,), {
  'DESCRIPTOR' : _REPLICACONFIG,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.ReplicaConfig)
  })
_sym_db.RegisterMessage(ReplicaConfig)

DeploymentInfo = _reflection.GeneratedProtocolMessageType('DeploymentInfo', (_message.Message,), {
  'DESCRIPTOR' : _DEPLOYMENTINFO,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.DeploymentInfo)
  })
_sym_db.RegisterMessage(DeploymentInfo)

DeploymentRoute = _reflection.GeneratedProtocolMessageType('DeploymentRoute', (_message.Message,), {
  'DESCRIPTOR' : _DEPLOYMENTROUTE,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.DeploymentRoute)
  })
_sym_db.RegisterMessage(DeploymentRoute)

DeploymentRouteList = _reflection.GeneratedProtocolMessageType('DeploymentRouteList', (_message.Message,), {
  'DESCRIPTOR' : _DEPLOYMENTROUTELIST,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.DeploymentRouteList)
  })
_sym_db.RegisterMessage(DeploymentRouteList)

DeploymentStatusInfo = _reflection.GeneratedProtocolMessageType('DeploymentStatusInfo', (_message.Message,), {
  'DESCRIPTOR' : _DEPLOYMENTSTATUSINFO,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.DeploymentStatusInfo)
  })
_sym_db.RegisterMessage(DeploymentStatusInfo)

DeploymentStatusInfoList = _reflection.GeneratedProtocolMessageType('DeploymentStatusInfoList', (_message.Message,), {
  'DESCRIPTOR' : _DEPLOYMENTSTATUSINFOLIST,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.DeploymentStatusInfoList)
  })
_sym_db.RegisterMessage(DeploymentStatusInfoList)

ApplicationStatusInfo = _reflection.GeneratedProtocolMessageType('ApplicationStatusInfo', (_message.Message,), {
  'DESCRIPTOR' : _APPLICATIONSTATUSINFO,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.ApplicationStatusInfo)
  })
_sym_db.RegisterMessage(ApplicationStatusInfo)

StatusOverview = _reflection.GeneratedProtocolMessageType('StatusOverview', (_message.Message,), {
  'DESCRIPTOR' : _STATUSOVERVIEW,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.StatusOverview)
  })
_sym_db.RegisterMessage(StatusOverview)

ListApplicationsRequest = _reflection.GeneratedProtocolMessageType('ListApplicationsRequest', (_message.Message,), {
  'DESCRIPTOR' : _LISTAPPLICATIONSREQUEST,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.ListApplicationsRequest)
  })
_sym_db.RegisterMessage(ListApplicationsRequest)

ListApplicationsResponse = _reflection.GeneratedProtocolMessageType('ListApplicationsResponse', (_message.Message,), {
  'DESCRIPTOR' : _LISTAPPLICATIONSRESPONSE,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.ListApplicationsResponse)
  })
_sym_db.RegisterMessage(ListApplicationsResponse)

HealthzRequest = _reflection.GeneratedProtocolMessageType('HealthzRequest', (_message.Message,), {
  'DESCRIPTOR' : _HEALTHZREQUEST,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.HealthzRequest)
  })
_sym_db.RegisterMessage(HealthzRequest)

HealthzResponse = _reflection.GeneratedProtocolMessageType('HealthzResponse', (_message.Message,), {
  'DESCRIPTOR' : _HEALTHZRESPONSE,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.HealthzResponse)
  })
_sym_db.RegisterMessage(HealthzResponse)

UserDefinedMessage = _reflection.GeneratedProtocolMessageType('UserDefinedMessage', (_message.Message,), {
  'DESCRIPTOR' : _USERDEFINEDMESSAGE,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.UserDefinedMessage)
  })
_sym_db.RegisterMessage(UserDefinedMessage)

UserDefinedResponse = _reflection.GeneratedProtocolMessageType('UserDefinedResponse', (_message.Message,), {
  'DESCRIPTOR' : _USERDEFINEDRESPONSE,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.UserDefinedResponse)
  })
_sym_db.RegisterMessage(UserDefinedResponse)

UserDefinedMessage2 = _reflection.GeneratedProtocolMessageType('UserDefinedMessage2', (_message.Message,), {
  'DESCRIPTOR' : _USERDEFINEDMESSAGE2,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.UserDefinedMessage2)
  })
_sym_db.RegisterMessage(UserDefinedMessage2)

UserDefinedResponse2 = _reflection.GeneratedProtocolMessageType('UserDefinedResponse2', (_message.Message,), {
  'DESCRIPTOR' : _USERDEFINEDRESPONSE2,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.UserDefinedResponse2)
  })
_sym_db.RegisterMessage(UserDefinedResponse2)

FruitAmounts = _reflection.GeneratedProtocolMessageType('FruitAmounts', (_message.Message,), {
  'DESCRIPTOR' : _FRUITAMOUNTS,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.FruitAmounts)
  })
_sym_db.RegisterMessage(FruitAmounts)

FruitCosts = _reflection.GeneratedProtocolMessageType('FruitCosts', (_message.Message,), {
  'DESCRIPTOR' : _FRUITCOSTS,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.FruitCosts)
  })
_sym_db.RegisterMessage(FruitCosts)

ArrayData = _reflection.GeneratedProtocolMessageType('ArrayData', (_message.Message,), {
  'DESCRIPTOR' : _ARRAYDATA,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.ArrayData)
  })
_sym_db.RegisterMessage(ArrayData)

StringData = _reflection.GeneratedProtocolMessageType('StringData', (_message.Message,), {
  'DESCRIPTOR' : _STRINGDATA,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.StringData)
  })
_sym_db.RegisterMessage(StringData)

ModelOutput = _reflection.GeneratedProtocolMessageType('ModelOutput', (_message.Message,), {
  'DESCRIPTOR' : _MODELOUTPUT,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.ModelOutput)
  })
_sym_db.RegisterMessage(ModelOutput)

DeploymentArgs = _reflection.GeneratedProtocolMessageType('DeploymentArgs', (_message.Message,), {
  'DESCRIPTOR' : _DEPLOYMENTARGS,
  '__module__' : 'ray.serve.generated.serve_pb2'
  # @@protoc_insertion_point(class_scope:ray.serve.DeploymentArgs)
  })
_sym_db.RegisterMessage(DeploymentArgs)

_RAYSERVEAPISERVICE = DESCRIPTOR.services_by_name['RayServeAPIService']
_USERDEFINEDSERVICE = DESCRIPTOR.services_by_name['UserDefinedService']
_FRUITSERVICE = DESCRIPTOR.services_by_name['FruitService']
_RAYSERVEBENCHMARKSERVICE = DESCRIPTOR.services_by_name['RayServeBenchmarkService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\026io.ray.serve.generatedB\013ServeProtosP\001\370\001\001'
  _REQUESTMETADATA_CONTEXTENTRY._options = None
  _REQUESTMETADATA_CONTEXTENTRY._serialized_options = b'8\001'
  _LONGPOLLREQUEST_KEYSTOSNAPSHOTIDSENTRY._options = None
  _LONGPOLLREQUEST_KEYSTOSNAPSHOTIDSENTRY._serialized_options = b'8\001'
  _LONGPOLLRESULT_UPDATEDOBJECTSENTRY._options = None
  _LONGPOLLRESULT_UPDATEDOBJECTSENTRY._serialized_options = b'8\001'
  _ENDPOINTINFO_CONFIGENTRY._options = None
  _ENDPOINTINFO_CONFIGENTRY._serialized_options = b'8\001'
  _ENDPOINTSET_ENDPOINTSENTRY._options = None
  _ENDPOINTSET_ENDPOINTSENTRY._serialized_options = b'8\001'
  _ENCODINGTYPE._serialized_start=6080
  _ENCODINGTYPE._serialized_end=6114
  _DEPLOYMENTLANGUAGE._serialized_start=6116
  _DEPLOYMENTLANGUAGE._serialized_end=6158
  _TARGETCAPACITYDIRECTION._serialized_start=6160
  _TARGETCAPACITYDIRECTION._serialized_end=6214
  _DEPLOYMENTSTATUS._serialized_start=6217
  _DEPLOYMENTSTATUS._serialized_end=6399
  _DEPLOYMENTSTATUSTRIGGER._serialized_start=6402
  _DEPLOYMENTSTATUSTRIGGER._serialized_end=6912
  _APPLICATIONSTATUS._serialized_start=6915
  _APPLICATIONSTATUS._serialized_end=7141
  _AUTOSCALINGCONFIG._serialized_start=44
  _AUTOSCALINGCONFIG._serialized_end=888
  _LOGGINGCONFIG._serialized_start=891
  _LOGGINGCONFIG._serialized_end=1059
  _DEPLOYMENTCONFIG._serialized_start=1062
  _DEPLOYMENTCONFIG._serialized_end=1836
  _REQUESTMETADATA._serialized_start=1839
  _REQUESTMETADATA._serialized_end=2195
  _REQUESTMETADATA_CONTEXTENTRY._serialized_start=2137
  _REQUESTMETADATA_CONTEXTENTRY._serialized_end=2195
  _REQUESTWRAPPER._serialized_start=2197
  _REQUESTWRAPPER._serialized_end=2233
  _UPDATEDOBJECT._serialized_start=2235
  _UPDATEDOBJECT._serialized_end=2324
  _LONGPOLLREQUEST._serialized_start=2327
  _LONGPOLLREQUEST._serialized_end=2514
  _LONGPOLLREQUEST_KEYSTOSNAPSHOTIDSENTRY._serialized_start=2446
  _LONGPOLLREQUEST_KEYSTOSNAPSHOTIDSENTRY._serialized_end=2514
  _LONGPOLLRESULT._serialized_start=2517
  _LONGPOLLRESULT._serialized_end=2714
  _LONGPOLLRESULT_UPDATEDOBJECTSENTRY._serialized_start=2623
  _LONGPOLLRESULT_UPDATEDOBJECTSENTRY._serialized_end=2714
  _ENDPOINTINFO._serialized_start=2717
  _ENDPOINTINFO._serialized_end=2910
  _ENDPOINTINFO_CONFIGENTRY._serialized_start=2853
  _ENDPOINTINFO_CONFIGENTRY._serialized_end=2910
  _ENDPOINTSET._serialized_start=2913
  _ENDPOINTSET._serialized_end=3082
  _ENDPOINTSET_ENDPOINTSENTRY._serialized_start=2997
  _ENDPOINTSET_ENDPOINTSENTRY._serialized_end=3082
  _ACTORNAMELIST._serialized_start=3084
  _ACTORNAMELIST._serialized_end=3121
  _DEPLOYMENTVERSION._serialized_start=3124
  _DEPLOYMENTVERSION._serialized_end=3461
  _REPLICACONFIG._serialized_start=3464
  _REPLICACONFIG._serialized_end=3837
  _DEPLOYMENTINFO._serialized_start=3840
  _DEPLOYMENTINFO._serialized_end=4277
  _DEPLOYMENTROUTE._serialized_start=4279
  _DEPLOYMENTROUTE._serialized_end=4386
  _DEPLOYMENTROUTELIST._serialized_start=4388
  _DEPLOYMENTROUTELIST._serialized_end=4482
  _DEPLOYMENTSTATUSINFO._serialized_start=4485
  _DEPLOYMENTSTATUSINFO._serialized_end=4681
  _DEPLOYMENTSTATUSINFOLIST._serialized_start=4683
  _DEPLOYMENTSTATUSINFOLIST._serialized_end=4798
  _APPLICATIONSTATUSINFO._serialized_start=4801
  _APPLICATIONSTATUSINFO._serialized_end=4955
  _STATUSOVERVIEW._serialized_start=4958
  _STATUSOVERVIEW._serialized_end=5145
  _LISTAPPLICATIONSREQUEST._serialized_start=5147
  _LISTAPPLICATIONSREQUEST._serialized_end=5172
  _LISTAPPLICATIONSRESPONSE._serialized_start=5174
  _LISTAPPLICATIONSRESPONSE._serialized_end=5245
  _HEALTHZREQUEST._serialized_start=5247
  _HEALTHZREQUEST._serialized_end=5263
  _HEALTHZRESPONSE._serialized_start=5265
  _HEALTHZRESPONSE._serialized_end=5308
  _USERDEFINEDMESSAGE._serialized_start=5310
  _USERDEFINEDMESSAGE._serialized_end=5386
  _USERDEFINEDRESPONSE._serialized_start=5388
  _USERDEFINEDRESPONSE._serialized_end=5460
  _USERDEFINEDMESSAGE2._serialized_start=5462
  _USERDEFINEDMESSAGE2._serialized_end=5483
  _USERDEFINEDRESPONSE2._serialized_start=5485
  _USERDEFINEDRESPONSE2._serialized_end=5535
  _FRUITAMOUNTS._serialized_start=5537
  _FRUITAMOUNTS._serialized_end=5621
  _FRUITCOSTS._serialized_start=5623
  _FRUITCOSTS._serialized_end=5657
  _ARRAYDATA._serialized_start=5659
  _ARRAYDATA._serialized_end=5690
  _STRINGDATA._serialized_start=5692
  _STRINGDATA._serialized_end=5724
  _MODELOUTPUT._serialized_start=5726
  _MODELOUTPUT._serialized_end=5763
  _DEPLOYMENTARGS._serialized_start=5766
  _DEPLOYMENTARGS._serialized_end=6078
  _RAYSERVEAPISERVICE._serialized_start=7144
  _RAYSERVEAPISERVICE._serialized_end=7323
  _USERDEFINEDSERVICE._serialized_start=7326
  _USERDEFINEDSERVICE._serialized_end=7649
  _FRUITSERVICE._serialized_start=7651
  _FRUITSERVICE._serialized_end=7727
  _RAYSERVEBENCHMARKSERVICE._serialized_start=7730
  _RAYSERVEBENCHMARKSERVICE._serialized_end=7882
# @@protoc_insertion_point(module_scope)
