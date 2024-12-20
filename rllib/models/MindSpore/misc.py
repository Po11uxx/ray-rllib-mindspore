""" Code adapted from https://github.com/ikostrikov/pytorch-a3c"""
import numpy as np
from typing import Union, Tuple, Any, List

from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_mindspore
from ray.rllib.utils.typing import TensorType

mindspore, nn = try_import_mindspore()

import mindspore
from mindspore import ops
from mindspore.common import initializer
from mindspore.common.initializer import Constant
from mindspore import Parameter

@DeveloperAPI
def normc_initializer(std: float = 1.0):
    def initializer(parameter: Parameter):
        # print("Parameter shape:", parameter.shape)
        # 计算每个神经元的L2范数
        norm = ops.Sqrt()(ops.ReduceSum()(parameter ** 2, 1))

        # print("Norm value:", norm)

        # 将范数的形状调整为 (parameter.shape[0], 1) 以进行广播
        norm = ops.Reshape()(norm, (parameter.shape[0], 1))

        # print("Reshaped norm:", norm.shape)

        scaled_parameter = (parameter * std) / norm

        # print("Scaled parameter:", scaled_parameter)
        parameter.set_data(scaled_parameter)

        # 标准化并缩放
        # parameter.set_data((parameter * std) / norm)

    return initializer


# @DeveloperAPI
# def normc_initializer(std: float = 1.0) -> Any:
#     def initializer(tensor):
#         tensor.data.norm(0, 1)
#         tensor.data *= std / mindspore.ops.sqrt(tensor.data.pow(2).sum(1, keepdims=True))
#
#     return initializer


@DeveloperAPI
def same_padding(
    in_size: Tuple[int, int],
    filter_size: Union[int, Tuple[int, int]],
    stride_size: Union[int, Tuple[int, int]],
) -> (Union[int, Tuple[int, int]], Tuple[int, int]):
    """Note: Padding is added to match TF conv2d `same` padding.

    See www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution

    Args:
        in_size: Rows (Height), Column (Width) for input
        stride_size (Union[int,Tuple[int, int]]): Rows (Height), column (Width)
            for stride. If int, height == width.
        filter_size: Rows (Height), column (Width) for filter

    Returns:
        padding: For input into mindspore.nn.ZeroPad2d.
        output: Output shape after padding and convolution.
    """
    in_height, in_width = in_size
    if isinstance(filter_size, int):
        filter_height, filter_width = filter_size, filter_size
    else:
        filter_height, filter_width = filter_size
    if isinstance(stride_size, (int, float)):
        stride_height, stride_width = int(stride_size), int(stride_size)
    else:
        stride_height, stride_width = int(stride_size[0]), int(stride_size[1])

    out_height = int(np.ceil(float(in_height) / float(stride_height)))
    out_width = int(np.ceil(float(in_width) / float(stride_width)))

    pad_along_height = int((out_height - 1) * stride_height + filter_height - in_height)
    pad_along_width = int((out_width - 1) * stride_width + filter_width - in_width)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    output = (out_height, out_width)
    return padding, output


@DeveloperAPI
def same_padding_transpose_after_stride(
    strided_size: Tuple[int, int],
    kernel: Tuple[int, int],
    stride: Union[int, Tuple[int, int]],
) -> (Union[int, Tuple[int, int]], Tuple[int, int]):
    """Computes padding and output size such that TF Conv2DTranspose `same` is matched.

    Note that when padding="same", TensorFlow's Conv2DTranspose makes sure that
    0-padding is added to the already strided image in such a way that the output image
    has the same size as the input image times the stride (and no matter the
    kernel size).

    For example: Input image is (4, 4, 24) (not yet strided), padding is "same",
    stride=2, kernel=5.

    First, the input image is strided (with stride=2):

    Input image (4x4):
    A B C D
    E F G H
    I J K L
    M N O P

    Stride with stride=2 -> (7x7)
    A 0 B 0 C 0 D
    0 0 0 0 0 0 0
    E 0 F 0 G 0 H
    0 0 0 0 0 0 0
    I 0 J 0 K 0 L
    0 0 0 0 0 0 0
    M 0 N 0 O 0 P

    Then this strided image (strided_size=7x7) is padded (exact padding values will be
    output by this function):

    padding -> (left=3, right=2, top=3, bottom=2)

    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 A 0 B 0 C 0 D 0 0
    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 E 0 F 0 G 0 H 0 0
    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 I 0 J 0 K 0 L 0 0
    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 M 0 N 0 O 0 P 0 0
    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0

    Then deconvolution with kernel=5 yields an output image of 8x8 (x num output
    filters).

    Args:
        strided_size: The size (width x height) of the already strided image.
        kernel: Either width x height (tuple of ints) or - if a square kernel is used -
            a single int for both width and height.
        stride: Either stride width x stride height (tuple of ints) or - if square
            striding is used - a single int for both width- and height striding.

    Returns:
        Tuple consisting of 1) `padding`: A 4-tuple to pad the input after(!) striding.
        The values are for left, right, top, and bottom padding, individually.
        This 4-tuple can be used in a mindspore.nn.ZeroPad2d layer, and 2) the output shape
        after striding, padding, and the conv transpose layer.
    """

    # Solve single int (squared) inputs for kernel and/or stride.
    k_w, k_h = (kernel, kernel) if isinstance(kernel, int) else kernel
    s_w, s_h = (stride, stride) if isinstance(stride, int) else stride

    # Compute the total size of the 0-padding on both axes. If results are odd numbers,
    # the padding on e.g. left and right (or top and bottom) side will have to differ
    # by 1.
    pad_total_w, pad_total_h = k_w - 1 + s_w - 1, k_h - 1 + s_h - 1
    pad_right = pad_total_w // 2
    pad_left = pad_right + (1 if pad_total_w % 2 == 1 else 0)
    pad_bottom = pad_total_h // 2
    pad_top = pad_bottom + (1 if pad_total_h % 2 == 1 else 0)

    # Compute the output size.
    output_shape = (
        strided_size[0] + pad_total_w - k_w + 1,
        strided_size[1] + pad_total_h - k_h + 1,
    )

    # Return padding and output shape.
    return (pad_left, pad_right, pad_top, pad_bottom), output_shape


@DeveloperAPI
def valid_padding(
    in_size: Tuple[int, int],
    filter_size: Union[int, Tuple[int, int]],
    stride_size: Union[int, Tuple[int, int]],
) -> Tuple[int, int]:
    """Emulates TF Conv2DLayer "valid" padding (no padding) and computes output dims.

    This method, analogous to its "same" counterpart, but it only computes the output
    image size, since valid padding means (0, 0, 0, 0).

    See www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution

    Args:
        in_size: Rows (Height), Column (Width) for input
        stride_size (Union[int,Tuple[int, int]]): Rows (Height), column (Width)
            for stride. If int, height == width.
        filter_size: Rows (Height), column (Width) for filter

    Returns:
        The output shape after padding and convolution.
    """
    in_height, in_width = in_size
    if isinstance(filter_size, int):
        filter_height, filter_width = filter_size, filter_size
    else:
        filter_height, filter_width = filter_size
    if isinstance(stride_size, (int, float)):
        stride_height, stride_width = int(stride_size), int(stride_size)
    else:
        stride_height, stride_width = int(stride_size[0]), int(stride_size[1])

    out_height = int(np.ceil((in_height - filter_height + 1) / float(stride_height)))
    out_width = int(np.ceil((in_width - filter_width + 1) / float(stride_width)))

    return (out_height, out_width)


@DeveloperAPI
class SlimConv2d(nn.Cell):
    """Simple mock of tf.slim Conv2d"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        # Defaulting these to nn.[..] will break soft mindspore import.
        initializer: Any = "default",
        activation_fn: Any = "default",
        bias_init: float = 0,
    ):
        """Creates a standard Conv2d layer, similar to mindspore.nn.Conv2d

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel: If int, the kernel is
                a tuple(x,x). Elsewise, the tuple can be specified
            stride: Controls the stride
                for the cross-correlation. If int, the stride is a
                tuple(x,x). Elsewise, the tuple can be specified
            padding: Controls the amount
                of implicit zero-paddings during the conv operation
            initializer: Initializer function for kernel weights
            activation_fn: Activation function at the end of layer
            bias_init: Initalize bias weights to bias_init const
        """
        super(SlimConv2d, self).__init__()
        layers = []
        # Padding layer.
        if padding:
            layers.append(nn.ZeroPad2d(padding))
        # Actual Conv2D layer (including correct initialization logic).
        conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        if initializer:
            if initializer == "default":
                #TODO: to be determined
                initializer = nn.init.xavier_uniform
            initializer(conv.weight)
        nn.init.constant_(conv.bias, bias_init)
        layers.append(conv)
        # Activation function (if any; default=ReLu).
        if isinstance(activation_fn, str):
            if activation_fn == "default":
                activation_fn = nn.ReLU
            else:
                activation_fn = get_activation_fn(activation_fn, "mindspore")
        if activation_fn is not None:
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.SequentialCell(*layers)

    def construct(self, x: TensorType) -> TensorType:
        return self._model(x)


@DeveloperAPI
class SlimFC(nn.Cell):
    """Simple PyTorch version of `linear` function"""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        initializer: Any = None,
        activation_fn: Any = None,
        use_bias: bool = True,
        bias_init: float = 0.0,
    ):
        """Creates a standard FC layer, similar to mindspore.nn.Dense

        Args:
            in_size: Input size for FC Layer
            out_size: Output size for FC Layer
            initializer: Initializer function for FC layer weights
            activation_fn: Activation function at the end of layer
            use_bias: Whether to add bias weights or not
            bias_init: Initalize bias weights to bias_init const
        """
        super(SlimFC, self).__init__()
        layers = []
        # Actual nn.Dense layer (including correct initialization logic).
        linear = nn.Dense(in_size, out_size, has_bias=use_bias)
        if initializer is None:
            # TODO: to be determined
            initializer = nn.init.xavier_uniform
        initializer(linear.weight)
        if use_bias is True:
            # nn.init.constant_
            linear.bias.set_data(mindspore.numpy.zeros(linear.bias.shape))
        layers.append(linear)
        # Activation function (if any; default=None (linear)).
        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn, "mindspore")
        if activation_fn is not None:
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.SequentialCell(*layers)

    def construct(self, x: TensorType) -> TensorType:
        return self._model(x)


@DeveloperAPI
class AppendBiasLayer(nn.Cell):
    """Simple bias appending layer for free_log_std."""

    def __init__(self, num_bias_vars: int):
        super().__init__()
        self.log_std = mindspore.Parameter(mindspore.ops.is_tensor([0.0] * num_bias_vars))
        self.register_parameter("log_std", self.log_std)

    def construct(self, x: TensorType) -> TensorType:
        out = mindspore.cat([x, self.log_std.unsqueeze(0).repeat([len(x), 1])], axis=1)
        return out


@DeveloperAPI
class Reshape(nn.Cell):
    """Standard module that reshapes/views a tensor"""

    def __init__(self, shape: List):
        super().__init__()
        self.shape = shape

    def construct(self, x):
        return x.view(*self.shape)
