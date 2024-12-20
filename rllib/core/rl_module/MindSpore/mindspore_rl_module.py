from typing import Any, Collection, Dict, Optional, Union, Type

import gymnasium as gym
from packaging import version

from ray.rllib.core.rl_module.apis import InferenceOnlyAPI
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.MindSpore.mindspore_compile_config import MindSporeCompileConfig
from ray.rllib.models.MindSpore.mindspore_distributions import (
    MindSporeCategorical,
    MindSporeDiagGaussian,
    MindSporeDistribution,
)
from ray.rllib.utils.annotations import override, OverrideToImplementCustomLogic
from ray.rllib.utils.framework import try_import_mindspore
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.mindspore_utils import (
    convert_to_mindspore_tensor,
    MINDSPORE_COMPILE_REQUIRED_VERSION,
)
from ray.rllib.utils.typing import StateDict

mindspore, nn = try_import_mindspore()


class MindSporeRLModule(nn.Cell, RLModule):
    """A base class for RLlib PyMindSpore RLModules.

    Note that the `_forward` methods of this class can be 'mindspore.compiled' individually:
        - `MindSporeRLModule._forward_train()`
        - `MindSporeRLModule._forward_inference()`
        - `MindSporeRLModule._forward_exploration()`

    As a rule of thumb, they should only contain mindspore-native tensor manipulations,
    or otherwise they may yield wrong outputs. In particular, the creation of RLlib
    distributions inside these methods should be avoided when using `mindspore.compile`.
    When in doubt, you can use `mindspore.dynamo.explain()` to check whether a compiled
    method has broken up into multiple sub-graphs.

    Compiling these methods can bring speedups under certain conditions.
    """

    framework: str = "mindspore"

    # Stick with mindspore default.
    STATE_FILE_NAME = "module_state.pt"

    def __init__(self, *args, **kwargs) -> None:
        nn.Cell.__init__(self)
        RLModule.__init__(self, *args, **kwargs)

        # If an inference-only class AND self.inference_only is True,
        # remove all attributes that are returned by
        # `self.get_non_inference_attributes()`.
        if self.inference_only and isinstance(self, InferenceOnlyAPI):
            for attr in self.get_non_inference_attributes():
                parts = attr.split(".")
                if not hasattr(self, parts[0]):
                    continue
                target = getattr(self, parts[0])
                # Traverse from the next part on (if nested).
                for part in parts[1:]:
                    if not hasattr(target, part):
                        target = None
                        break
                    target = getattr(target, part)
                # Delete, if target is valid.
                if target is not None:
                    del target

    def compile(self, compile_config: MindSporeCompileConfig):
        """Compile the forward methods of this module.

        This is a convenience method that calls `compile_wrapper` with the given
        compile_config.

        Args:
            compile_config: The compile config to use.
        """
        return compile_wrapper(self, compile_config)

    @OverrideToImplementCustomLogic
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # By default, calls the generic `_forward()` method, but with a no-grad context
        # for performance reasons.
        with mindspore.no_grad():
            return self._forward(batch, **kwargs)

    @OverrideToImplementCustomLogic
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # By default, calls the generic `_forward()` method, but with a no-grad context
        # for performance reasons.
        with mindspore.no_grad():
            return self._forward(batch, **kwargs)

    @OverrideToImplementCustomLogic
    @override(RLModule)
    def get_state(
        self,
        components: Optional[Union[str, Collection[str]]] = None,
        *,
        not_components: Optional[Union[str, Collection[str]]] = None,
        inference_only: bool = False,
        **kwargs,
    ) -> StateDict:
        state_dict = self.state_dict()
        # Filter out `inference_only` keys from the state dict if `inference_only` and
        # this RLModule is NOT `inference_only` (but does implement the
        # InferenceOnlyAPI).
        if (
            inference_only
            and not self.inference_only
            and isinstance(self, InferenceOnlyAPI)
        ):
            attr = self.get_non_inference_attributes()
            for key in list(state_dict.keys()):
                if any(
                    key.startswith(a) and (len(key) == len(a) or key[len(a)] == ".")
                    for a in attr
                ):
                    del state_dict[key]
        return convert_to_numpy(state_dict)

    @OverrideToImplementCustomLogic
    @override(RLModule)
    def set_state(self, state: StateDict) -> None:
        # If state contains more keys than `self.state_dict()`, then we simply ignore
        # these keys (strict=False). This is most likely due to `state` coming from
        # an `inference_only=False` RLModule, while `self` is an `inference_only=True`
        # RLModule.
        self.load_state_dict(convert_to_mindspore_tensor(state), strict=False)

    @OverrideToImplementCustomLogic
    @override(RLModule)
    def get_inference_action_dist_cls(self) -> Type[MindSporeDistribution]:
        if self.action_dist_cls is not None:
            return self.action_dist_cls
        elif isinstance(self.action_space, gym.spaces.Discrete):
            return MindSporeCategorical
        elif isinstance(self.action_space, gym.spaces.Box):
            return MindSporeDiagGaussian
        else:
            raise ValueError(
                f"Default action distribution for action space "
                f"{self.action_space} not supported! Either set the "
                f"`self.action_dist_cls` property in your RLModule's `setup()` method "
                f"to a subclass of `ray.rllib.models.mindspore.mindspore_distributions."
                f"MindSporeDistribution` or - if you need different distributions for "
                f"inference and training - override the three methods: "
                f"`get_inference_action_dist_cls`, `get_exploration_action_dist_cls`, "
                f"and `get_train_action_dist_cls` in your RLModule."
            )

    @OverrideToImplementCustomLogic
    @override(RLModule)
    def get_exploration_action_dist_cls(self) -> Type[MindSporeDistribution]:
        return self.get_inference_action_dist_cls()

    @OverrideToImplementCustomLogic
    @override(RLModule)
    def get_train_action_dist_cls(self) -> Type[MindSporeDistribution]:
        return self.get_inference_action_dist_cls()

    @override(nn.Cell)
    def construct(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """DO NOT OVERRIDE!

        This is aliased to `self.forward_train` because MindSpore DDP requires a forward
        method to be implemented for backpropagation to work.

        Instead, override:
        `_forward()` to define a generic forward pass for all phases (exploration,
        inference, training)
        `_forward_inference()` to define the forward pass for action inference in
        deployment/production (no exploration).
        `_forward_exploration()` to define the forward pass for action inference during
        training sample collection (w/ exploration behavior).
        `_forward_train()` to define the forward pass prior to loss computation.
        """
        return self.forward_train(batch, **kwargs)


class MindSporeDDPRLModule(RLModule, mindspore.ParallelMode):
    def __init__(self, *args, **kwargs) -> None:
        mindspore.ParallelMode.__init__(self, *args, **kwargs)
        # We do not want to call RLModule.__init__ here because all we need is
        # the interface of that base-class not the actual implementation.
        # RLModule.__init__(self, *args, **kwargs)
        self.observation_space = self.unwrapped().observation_space
        self.action_space = self.unwrapped().action_space
        self.inference_only = self.unwrapped().inference_only
        self.learner_only = self.unwrapped().learner_only
        self.model_config = self.unwrapped().model_config
        self.catalog = self.unwrapped().catalog

        # Deprecated.
        self.config = self.unwrapped().config

    @override(RLModule)
    def get_inference_action_dist_cls(self, *args, **kwargs) -> Type[MindSporeDistribution]:
        return self.unwrapped().get_inference_action_dist_cls(*args, **kwargs)

    @override(RLModule)
    def get_exploration_action_dist_cls(
        self, *args, **kwargs
    ) -> Type[MindSporeDistribution]:
        return self.unwrapped().get_exploration_action_dist_cls(*args, **kwargs)

    @override(RLModule)
    def get_train_action_dist_cls(self, *args, **kwargs) -> Type[MindSporeDistribution]:
        return self.unwrapped().get_train_action_dist_cls(*args, **kwargs)

    @override(RLModule)
    def get_initial_state(self) -> Any:
        return self.unwrapped().get_initial_state()

    @override(RLModule)
    def is_stateful(self) -> bool:
        return self.unwrapped().is_stateful()

    @override(RLModule)
    def _forward(self, *args, **kwargs):
        return self.unwrapped()._forward(*args, **kwargs)

    @override(RLModule)
    def _forward_inference(self, *args, **kwargs) -> Dict[str, Any]:
        return self.unwrapped()._forward_inference(*args, **kwargs)

    @override(RLModule)
    def _forward_exploration(self, *args, **kwargs) -> Dict[str, Any]:
        return self.unwrapped()._forward_exploration(*args, **kwargs)

    @override(RLModule)
    def _forward_train(self, *args, **kwargs):
        return self(*args, **kwargs)

    @override(RLModule)
    def get_state(self, *args, **kwargs):
        return self.unwrapped().get_state(*args, **kwargs)

    @override(RLModule)
    def set_state(self, *args, **kwargs):
        self.unwrapped().set_state(*args, **kwargs)

    @override(RLModule)
    def save_to_path(self, *args, **kwargs):
        self.unwrapped().save_to_path(*args, **kwargs)

    @override(RLModule)
    def restore_from_path(self, *args, **kwargs):
        self.unwrapped().restore_from_path(*args, **kwargs)

    @override(RLModule)
    def get_metadata(self, *args, **kwargs):
        self.unwrapped().get_metadata(*args, **kwargs)

    @override(RLModule)
    def unwrapped(self) -> "RLModule":
        return self.module


def compile_wrapper(rl_module: "MindSporeRLModule", compile_config: MindSporeCompileConfig):
    """A wrapper that compiles the forward methods of a MindSporeRLModule."""

    # TODO(Artur): Remove this once our requirements enforce mindspore >= 2.0.0
    # Check if mindspore framework supports mindspore.compile.
    if (
        mindspore is not None
        and version.parse(mindspore.__version__) < MINDSPORE_COMPILE_REQUIRED_VERSION
    ):
        raise ValueError("mindspore.compile is only supported from mindspore 2.0.0")

    compiled_forward_train = mindspore.compile(
        rl_module._forward_train,
        backend=compile_config.mindspore_dynamo_backend,
        mode=compile_config.mindspore_dynamo_mode,
        **compile_config.kwargs,
    )

    rl_module._forward_train = compiled_forward_train

    compiled_forward_inference = mindspore.compile(
        rl_module._forward_inference,
        backend=compile_config.mindspore_dynamo_backend,
        mode=compile_config.mindspore_dynamo_mode,
        **compile_config.kwargs,
    )

    rl_module._forward_inference = compiled_forward_inference

    compiled_forward_exploration = mindspore.compile(
        rl_module._forward_exploration,
        backend=compile_config.mindspore_dynamo_backend,
        mode=compile_config.mindspore_dynamo_mode,
        **compile_config.kwargs,
    )

    rl_module._forward_exploration = compiled_forward_exploration

    return rl_module
