from ray._private.ray_constants import env_integer

NODE_STATS_UPDATE_INTERVAL_SECONDS = env_integer(
    "NODE_STATS_UPDATE_INTERVAL_SECONDS", 15
)
RAY_DASHBOARD_HEAD_NODE_REGISTRATION_TIMEOUT = env_integer(
    "RAY_DASHBOARD_HEAD_NODE_REGISTRATION_TIMEOUT", 10
)
MAX_COUNT_OF_GCS_RPC_ERROR = 10
# This is consistent with gcs_node_manager.cc
MAX_DEAD_NODES_TO_CACHE = env_integer("RAY_maximum_gcs_dead_node_cached_count", 1000)
RAY_DASHBOARD_NODE_SUBSCRIBER_POLL_SIZE = env_integer(
    "RAY_DASHBOARD_NODE_SUBSCRIBER_POLL_SIZE", 200
)
RAY_DASHBOARD_AGENT_POLL_INTERVAL_S = env_integer(
    "RAY_DASHBOARD_AGENT_POLL_INTERVAL_S", 1
)