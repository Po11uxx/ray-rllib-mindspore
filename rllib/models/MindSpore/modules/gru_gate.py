from ray.rllib.utils.annotations import OldAPIStack
from ray.rllib.utils.framework import try_import_mindspore
from ray.rllib.utils.typing import TensorType

mindspore, nn = try_import_mindspore()


@OldAPIStack
class GRUGate(nn.Cell):
    """Implements a gated recurrent unit for use in AttentionNet"""

    def __init__(self, dim: int, init_bias: int = 0.0, **kwargs):
        """
        input_shape (mindspore.Tensor): dimension of the input
        init_bias: Bias added to every input to stabilize training
        """
        super().__init__(**kwargs)
        # Xavier initialization of mindspore tensors
        self._w_r = mindspore.Parameter(mindspore.ops.zeros(dim, dim))
        self._w_z = mindspore.Parameter(mindspore.ops.zeros(dim, dim))
        self._w_h = mindspore.Parameter(mindspore.ops.zeros(dim, dim))
        # TODO: to be determined
        nn.init.xavier_uniform_(self._w_r)
        nn.init.xavier_uniform_(self._w_z)
        nn.init.xavier_uniform_(self._w_h)
        self.register_parameter("_w_r", self._w_r)
        self.register_parameter("_w_z", self._w_z)
        self.register_parameter("_w_h", self._w_h)

        self._u_r = mindspore.Parameter(mindspore.ops.zeros(dim, dim))
        self._u_z = mindspore.Parameter(mindspore.ops.zeros(dim, dim))
        self._u_h = mindspore.Parameter(mindspore.ops.zeros(dim, dim))
        nn.init.xavier_uniform_(self._u_r)
        nn.init.xavier_uniform_(self._u_z)
        nn.init.xavier_uniform_(self._u_h)
        self.register_parameter("_u_r", self._u_r)
        self.register_parameter("_u_z", self._u_z)
        self.register_parameter("_u_h", self._u_h)

        self._bias_z = mindspore.Parameter(
            mindspore.ops.zeros(
                dim,
            ).fill_(init_bias)
        )
        self.register_parameter("_bias_z", self._bias_z)

    def forward(self, inputs: TensorType, **kwargs) -> TensorType:
        # Pass in internal state first.
        h, X = inputs

        r = mindspore.numpy.tensordot(X, self._w_r, dims=1) + mindspore.numpy.tensordot(
            h, self._u_r, dims=1
        )
        r = mindspore.ops.sigmoid(r)

        z = (
            mindspore.numpy.tensordot(X, self._w_z, dims=1)
            + mindspore.numpy.tensordot(h, self._u_z, dims=1)
            - self._bias_z
        )
        z = mindspore.ops.sigmoid(z)

        h_next = mindspore.numpy.tensordot(X, self._w_h, dims=1) + mindspore.numpy.tensordot(
            (h * r), self._u_h, dims=1
        )
        h_next = mindspore.ops.tanh(h_next)

        return (1 - z) * h + z * h_next
