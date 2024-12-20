import logging
from typing import Dict, List, Type, Union

import ray
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.mindspore_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.policy.mindspore_policy_v2 import MindSporePolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_mindspore
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.mindspore_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.utils.typing import TensorType

mindspore, nn = try_import_mindspore()

logger = logging.getLogger(__name__)


class PPOMindSporePolicy(
    ValueNetworkMixin,
    LearningRateSchedule,
    EntropyCoeffSchedule,
    KLCoeffMixin,
    MindSporePolicyV2,
):
    """MindSpore policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.algorithms.ppo.ppo.PPOConfig().to_dict(), **config)
        validate_config(config)

        MindSporePolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)

        self._initialize_loss_from_dummy_batch()

    @override(MindSporePolicyV2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = mindspore.ops.Reshape()(mask, [-1])
            num_valid = mindspore.ops.sum(mask)

            def reduce_mean_valid(t):
                return mindspore.ops.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = mindspore.ops.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = mindspore.ops.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = mindspore.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        # surrogate_loss = mindspore.ops.min(
        #     train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        #     train_batch[Postprocessing.ADVANTAGES]
        #     * mindspore.ops.clamp(
        #         logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
        #     ),
        # )
        surrogate_loss = mindspore.ops.Minimum()(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            mindspore.ops.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ) * train_batch[Postprocessing.ADVANTAGES]
        )
        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = mindspore.ops.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = mindspore.ops.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = mindspore.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = mindspore.tensor(0.0).to(surrogate_loss.device)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_gradients_computed").
    @override(MindSporePolicyV2)
    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    @override(MindSporePolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": mindspore.ops.mean(
                    mindspore.ops.stack(self.get_tower_stats("total_loss"), axis=-1)
                ),
                "policy_loss": mindspore.ops.mean(
                    mindspore.ops.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": mindspore.ops.mean(
                    mindspore.ops.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": mindspore.ops.mean(
                    mindspore.ops.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "kl": mindspore.ops.mean(mindspore.ops.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": mindspore.ops.mean(
                    mindspore.ops.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
            }
        )

    @override(MindSporePolicyV2)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in mindspore (issue #6962).
        # TODO: no_grad still necessary?
        # with mindspore.no_grad():
        # print("sample_batch:", type(sample_batch))
        # print("other_agent_batches:", type(other_agent_batches))
        if other_agent_batches is None:
            other_agent_batches = ()

            # 使用stop_gradient来阻止梯度计算
            # 假设sample_batch对象中的Tensor字段存储在sample_batch.data中
        for key, value in sample_batch.items():
            if isinstance(value, mindspore.Tensor):
                sample_batch.data[key] = mindspore.ops.stop_gradient(value)
        return compute_gae_for_sample_batch(
            self, sample_batch, other_agent_batches, episode
        )
