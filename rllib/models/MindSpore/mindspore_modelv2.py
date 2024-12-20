import gymnasium as gym
from typing import Dict, List, Union

from ray.rllib.models.MS_Modelv2 import MS_ModelV2
from ray.rllib.utils.annotations import OldAPIStack, override
from ray.rllib.utils.framework import try_import_mindspore
from ray.rllib.utils.typing import ModelConfigDict, TensorType

_, nn = try_import_mindspore()


@OldAPIStack
class MindSporeModelV2(MS_ModelV2):
    """MindSpore version of ModelV2.

    Note that this class by itself is not a valid model unless you
    inherit from nn.Cell and implement forward() in a subclass."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        """Initialize a MindSpore.

        Here is an example implementation for a subclass
        ``MyModelClass(MindSporeModelV2, nn.Cell)``::

            def __init__(self, *args, **kwargs):
                MindSporeModelV2.__init__(self, *args, **kwargs)
                nn.Cell.__init__(self)
                self._hidden_layers = nn.Sequential(...)
                self._logits = ...
                self._value_branch = ...
        """
        if not isinstance(self, nn.Cell):
            raise ValueError(
                "Subclasses of MindSporeModelV2 must also inherit from "
                "nn.Cell, e.g., MyModel(MindSporeModelV2, nn.Cell)"
            )

        MS_ModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            framework="mindspore",
        )

        # Dict to store per multi-gpu tower stats into.
        # In PyTorch multi-GPU, we use a single TorchPolicy and copy
        # it's Model(s) n times (1 copy for each GPU). When computing the loss
        # on each tower, we cannot store the stats (e.g. `entropy`) inside the
        # policy object as this would lead to race conditions between the
        # different towers all accessing the same property at the same time.
        self.tower_stats = {}

    @override(MS_ModelV2)
    def variables(
        self, as_dict: bool = False
    ) -> Union[List[TensorType], Dict[str, TensorType]]:
        p = list(self.parameters())
        if as_dict:
            return {k: p[i] for i, k in enumerate(self.state_dict().keys())}
        return p

    @override(MS_ModelV2)
    def trainable_variables(
        self, as_dict: bool = False
    ) -> Union[List[TensorType], Dict[str, TensorType]]:
        if as_dict:
            return {
                k: v for k, v in self.variables(as_dict=True).items() if v.requires_grad
            }
        return [v for v in self.variables() if v.requires_grad]
