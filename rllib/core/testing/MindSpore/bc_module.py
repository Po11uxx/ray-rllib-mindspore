from typing import Any, Dict

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.models.MindSpore.mindspore_distributions import MindSporeCategorical
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.core.rl_module.MindSpore.mindspore_rl_module import MindSporeRLModule
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_mindspore

mindspore, nn = try_import_mindspore()


class DiscreteBCMindSporeModule(MindSporeRLModule):
    def setup(self):
        input_dim = self.observation_space.shape[0]
        hidden_dim = self.model_config["fcnet_hiddens"][0]
        output_dim = self.action_space.n

        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.input_dim = input_dim

    def get_train_action_dist_cls(self):
        return MindSporeCategorical

    def get_exploration_action_dist_cls(self):
        return MindSporeCategorical

    def get_inference_action_dist_cls(self):
        return MindSporeCategorical

    @override(RLModule)
    def output_specs_exploration(self) -> SpecType:
        return [Columns.ACTION_DIST_INPUTS]

    @override(RLModule)
    def output_specs_inference(self) -> SpecType:
        return [Columns.ACTION_DIST_INPUTS]

    @override(RLModule)
    def output_specs_train(self) -> SpecType:
        return [Columns.ACTION_DIST_INPUTS]

    @override(RLModule)
    def _forward_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # with mindspore.no_grad():
        #     return self._forward_train(batch)
        is_training = self.training
        # 设置为非训练模式，停止梯度计算
        self.set_train(False)
        try:
            # 执行前向传播
            return self._common_forward(batch)
        finally:
            # 恢复原来的训练模式
            self.set_train(is_training)

    @override(RLModule)
    def _forward_exploration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # with mindspore.no_grad():
        #     return self._forward_train(batch)
        is_training = self.training
        # 设置为非训练模式，停止梯度计算
        self.set_train(False)
        try:
            # 执行前向传播
            return self._common_forward(batch)
        finally:
            # 恢复原来的训练模式
            self.set_train(is_training)

    @override(RLModule)
    def _forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        action_logits = self.policy(batch["obs"])
        return {Columns.ACTION_DIST_INPUTS: action_logits}


class BCMindSporeRLModuleWithSharedGlobalEncoder(MindSporeRLModule):
    """An example of an RLModule that uses an encoder shared with other things.

    For example, we could consider a multi-agent case where for inference each agent
    needs to know the global state of the environment, as well as the local state of
    itself. For better representation learning we would like to share the encoder
    across all the modules. So this module simply accepts the encoder object as its
    input argument and uses it to encode the global state. The local state is passed
    through as is. The policy head is then a simple MLP that takes the concatenation of
    the global and local state as input and outputs the action logits.

    """

    def __init__(
        self,
        encoder: nn.Module,
        local_dim: int,
        hidden_dim: int,
        action_dim: int,
        config=None,
    ) -> None:
        super().__init__(config=config)

        self.encoder = encoder
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + local_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def get_train_action_dist_cls(self):
        return MindSporeCategorical

    def get_exploration_action_dist_cls(self):
        return MindSporeCategorical

    def get_inference_action_dist_cls(self):
        return MindSporeCategorical

    @override(RLModule)
    def _default_input_specs(self):
        return [("obs", "global"), ("obs", "local")]

    @override(RLModule)
    def _forward_inference(self, batch):
        # with mindspore.no_grad():
        #     return self._common_forward(batch)
        is_training = self.training
        # 设置为非训练模式，停止梯度计算
        self.set_train(False)
        try:
            # 执行前向传播
            return self._common_forward(batch)
        finally:
            # 恢复原来的训练模式
            self.set_train(is_training)
    @override(RLModule)
    def _forward_exploration(self, batch):
        # with mindspore.no_grad():
        #     return self._common_forward(batch)
        is_training = self.training
        # 设置为非训练模式，停止梯度计算
        self.set_train(False)
        try:
            # 执行前向传播
            return self._common_forward(batch)
        finally:
            # 恢复原来的训练模式
            self.set_train(is_training)

    @override(RLModule)
    def _forward_train(self, batch):
        return self._common_forward(batch)

    def _common_forward(self, batch):
        obs = batch["obs"]
        global_enc = self.encoder(obs["global"])
        policy_in = mindspore.ops.cat([global_enc, obs["local"]], dim=-1)
        action_logits = self.policy_head(policy_in)

        return {Columns.ACTION_DIST_INPUTS: action_logits}


class BCMindSporeMultiAgentModuleWithSharedEncoder(MultiRLModule):
    def setup(self):
        module_specs = self.config.modules
        module_spec = next(iter(module_specs.values()))
        global_dim = module_spec.observation_space["global"].shape[0]
        hidden_dim = module_spec.model_config_dict["fcnet_hiddens"][0]
        shared_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        rl_modules = {}
        for module_id, module_spec in module_specs.items():
            rl_modules[module_id] = module_spec.module_class(
                config=self.config.modules[module_id].get_rl_module_config(),
                encoder=shared_encoder,
                local_dim=module_spec.observation_space["local"].shape[0],
                hidden_dim=hidden_dim,
                action_dim=module_spec.action_space.n,
            )

        self._rl_modules = rl_modules

    def serialize(self):
        # TODO (Kourosh): Implement when needed.
        raise NotImplementedError

    def deserialize(self, data):
        # TODO (Kourosh): Implement when needed.
        raise NotImplementedError