from ray.rllib.core.columns import Columns


DEFAULT_AGENT_ID = "default_agent"
DEFAULT_POLICY_ID = "default_policy"
DEFAULT_MODULE_ID = DEFAULT_POLICY_ID  # TODO (sven): Change this to "default_module"

COMPONENT_ENV_RUNNER = "env_runner"
COMPONENT_ENV_TO_MODULE_CONNECTOR = "env_to_module_connector"
COMPONENT_EVAL_ENV_RUNNER = "eval_env_runner"
COMPONENT_LEARNER = "learner"
COMPONENT_LEARNER_GROUP = "learner_group"
COMPONENT_METRICS_LOGGER = "metrics_logger"
COMPONENT_MODULE_TO_ENV_CONNECTOR = "module_to_env_connector"
COMPONENT_MULTI_RL_MODULE_SPEC = "_multi_rl_module_spec"
COMPONENT_OPTIMIZER = "optimizer"
COMPONENT_RL_MODULE = "rl_module"


__all__ = [
    "Columns",
    "COMPONENT_ENV_RUNNER",
    "COMPONENT_ENV_TO_MODULE_CONNECTOR",
    "COMPONENT_EVAL_ENV_RUNNER",
    "COMPONENT_LEARNER",
    "COMPONENT_LEARNER_GROUP",
    "COMPONENT_METRICS_LOGGER",
    "COMPONENT_MODULE_TO_ENV_CONNECTOR",
    "COMPONENT_MULTI_RL_MODULE_SPEC",
    "COMPONENT_OPTIMIZER",
    "COMPONENT_RL_MODULE",
    "DEFAULT_AGENT_ID",
    "DEFAULT_MODULE_ID",
    "DEFAULT_POLICY_ID",
]