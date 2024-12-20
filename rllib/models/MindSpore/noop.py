from ray.rllib.models.MS_Modelv2 import MS_ModelV2
from ray.rllib.models.MindSpore.mindspore_modelV2 import MindSporeModelV2
from ray.rllib.utils.annotations import OldAPIStack, override


@OldAPIStack
class MindSporeNoopModel(MindSporeModelV2):
    """Trivial model that just returns the obs flattened.

    This is the model used if use_state_preprocessor=False."""

    @override(MS_ModelV2)
    def forward(self, input_dict, state, seq_lens):
        return input_dict["obs_flat"].float(), state
