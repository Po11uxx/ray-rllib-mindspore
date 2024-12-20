from ray.rllib.utils.annotations import OldAPIStack
from ray.rllib.utils.framework import try_import_mindspore
from ray.rllib.utils.typing import TensorType
from typing import Optional

mindspore, nn = try_import_mindspore()


@OldAPIStack
class SkipConnection(nn.Cell):
    """Skip connection layer.

    Adds the original input to the output (regular residual layer) OR uses
    input as hidden state input to a given fan_in_layer.
    """

    def __init__(
        self, layer: nn.Cell, fan_in_layer: Optional[nn.Cell] = None, **kwargs
    ):
        """Initializes a SkipConnection nn Cell object.

        Args:
            layer (nn.Cell): Any layer processing inputs.
            fan_in_layer (Optional[nn.Cell]): An optional
                layer taking two inputs: The original input and the output
                of `layer`.
        """
        super().__init__(**kwargs)
        self._layer = layer
        self._fan_in_layer = fan_in_layer

    def forward(self, inputs: TensorType, **kwargs) -> TensorType:
        # del kwargs
        outputs = self._layer(inputs, **kwargs)
        # Residual case, just add inputs to outputs.
        if self._fan_in_layer is None:
            outputs = outputs + inputs
        # Fan-in e.g. RNN: Call fan-in with `inputs` and `outputs`.
        else:
            # NOTE: In the GRU case, `inputs` is the state input.
            outputs = self._fan_in_layer((inputs, outputs))

        return outputs
