"""
[1] - Attention Is All You Need - Vaswani, Jones, Shazeer, Parmar,
      Uszkoreit, Gomez, Kaiser - Google Brain/Research, U Toronto - 2017.
      https://arxiv.org/pdf/1706.03762.pdf
"""
from ray.rllib.utils.framework import try_import_mindspore
from ray.rllib.models.MindSpore.misc import SlimFC
from ray.rllib.utils.annotations import OldAPIStack
from ray.rllib.utils.mindspore_utils import sequence_mask
from ray.rllib.utils.framework import TensorType

mindspore, nn = try_import_mindspore()


@OldAPIStack
class MultiHeadAttention(nn.Cell):
    """A multi-head attention layer described in [1]."""

    def __init__(
        self, in_dim: int, out_dim: int, num_heads: int, head_dim: int, **kwargs
    ):
        """
        in_dim: Dimension of input
        out_dim: Dimension of output
        num_heads: Number of attention heads
        head_dim: Output dimension of each attention head
        """
        super().__init__(**kwargs)

        # No bias or non-linearity.
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._qkv_layer = SlimFC(
            in_size=in_dim, out_size=3 * num_heads * head_dim, use_bias=False
        )

        self._linear_layer = SlimFC(
            in_size=num_heads * head_dim, out_size=out_dim, use_bias=False
        )

    def forward(self, inputs: TensorType) -> TensorType:
        L = list(inputs.size())[1]  # length of segment
        H = self._num_heads  # number of attention heads
        D = self._head_dim  # attention head dimension

        qkv = self._qkv_layer(inputs)

        queries, keys, values = mindspore.ops.chunk(input=qkv, chunks=3, dim=-1)
        queries = queries[:, -L:]  # only query based on the segment

        queries = mindspore.ops.reshape(queries, [-1, L, H, D])
        keys = mindspore.ops.reshape(keys, [-1, L, H, D])
        values = mindspore.ops.reshape(values, [-1, L, H, D])

        score = mindspore.ops.einsum("bihd,bjhd->bijh", queries, keys)
        score = score / D**0.5

        # causal mask of the same length as the sequence
        mask = sequence_mask(mindspore.ops.arange(1, L + 1), dtype=score.dtype)
        mask = mask[None, :, :, None]
        mask = mask.float()

        masked_score = score * mask + 1e30 * (mask - 1.0)
        wmat = mindspore.ops.softmax(masked_score, dim=2)

        out = mindspore.ops.einsum("bijh,bjhd->bihd", wmat, values)
        shape = list(out.size())[:2] + [H * D]
        #        temp = mindspore.cat(temp2, [H * D], dim=0)
        out = mindspore.ops.reshape(out, shape)
        return self._linear_layer(out)
