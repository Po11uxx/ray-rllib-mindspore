import numpy as np

from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_mindspore, TensorType

mindspore, nn = try_import_mindspore()


class NoisyLayer(nn.Cell):
    r"""A Layer that adds learnable Noise to some previous layer's outputs.

    Consists of:
    - a common dense layer: y = w^{T}x + b
    - a noisy layer: y = (w + \epsilon_w*\sigma_w)^{T}x +
        (b+\epsilon_b*\sigma_b)
    , where \epsilon are random variables sampled from factorized normal
    distributions and \sigma are trainable variables which are expected to
    vanish along the training procedure.
    """

    def __init__(
        self, in_size: int, out_size: int, sigma0: float, activation: str = "relu"
    ):
        """Initializes a NoisyLayer object.

        Args:
            in_size: Input size for Noisy Layer
            out_size: Output size for Noisy Layer
            sigma0: Initialization value for sigma_b (bias noise)
            activation: Non-linear activation for Noisy Layer
        """
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.sigma0 = sigma0
        self.activation = get_activation_fn(activation, framework="mindspore")
        if self.activation is not None:
            self.activation = self.activation()

        sigma_w = mindspore.Parameter(
            mindspore.Tensor.from_numpy(
                np.random.uniform(
                    low=-1.0 / np.sqrt(float(self.in_size)),
                    high=1.0 / np.sqrt(float(self.in_size)),
                    size=[self.in_size, out_size],
                )
            ).float()
        )
        self.register_parameter("sigma_w", sigma_w)
        sigma_b = mindspore.Parameter(
            mindspore.Tensor.from_numpy(
                np.full(
                    shape=[out_size], fill_value=sigma0 / np.sqrt(float(self.in_size))
                )
            ).float()
        )
        self.register_parameter("sigma_b", sigma_b)

        w = mindspore.Parameter(
            mindspore.Tensor.from_numpy(
                np.full(
                    shape=[self.in_size, self.out_size],
                    fill_value=6 / np.sqrt(float(in_size) + float(out_size)),
                )
            ).float()
        )
        self.register_parameter("w", w)
        b = mindspore.Parameter(mindspore.Tensor.from_numpy(np.zeros([out_size])).float())
        self.register_parameter("b", b)

    def forward(self, inputs: TensorType) -> TensorType:
        epsilon_in = self._f_epsilon(
            mindspore.ops.normal(
                mean=mindspore.ops.zeros([self.in_size]), std=mindspore.ops.ones([self.in_size])
            ).to(inputs.device)
        )
        epsilon_out = self._f_epsilon(
            mindspore.ops.normal(
                mean=mindspore.ops.zeros([self.out_size]), std=mindspore.ops.ones([self.out_size])
            ).to(inputs.device)
        )
        epsilon_w = mindspore.ops.matmul(
            mindspore.ops.unsqueeze(epsilon_in, -1), other=mindspore.ops.unsqueeze(epsilon_out, 0)
        )
        epsilon_b = epsilon_out

        action_activation = (
            mindspore.ops.matmul(inputs, self.w + self.sigma_w * epsilon_w)
            + self.b
            + self.sigma_b * epsilon_b
        )

        if self.activation is not None:
            action_activation = self.activation(action_activation)
        return action_activation

    def _f_epsilon(self, x: TensorType) -> TensorType:
        return mindspore.ops.sign(x) * mindspore.ops.pow(mindspore.ops.abs(x), 0.5)
