from collections import defaultdict
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Optional,
    Sequence,
    Tuple,
)

from ray.rllib.algorithms.algorithm_config import (
    AlgorithmConfig,
    MindSporeCompileWhatToCompile,
)
from ray.rllib.core.learner.learner import Learner
from ray.rllib.core.rl_module.multi_rl_module import (
    MultiRLModule,
    MultiRLModuleSpec,
)
from ray.rllib.core.rl_module.rl_module import (
    RLModule,
    RLModuleSpec,
)
from ray.rllib.core.rl_module.MindSpore.mindspore_rl_module import (
    MindSporeCompileConfig,
    MindSporeDDPRLModule,
    MindSporeRLModule,
)
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import (
    override,
    OverrideToImplementCustomLogic,
)
from ray.rllib.utils.framework import try_import_mindspore
from ray.rllib.utils.metrics import (
    ALL_MODULES,
    NUM_TRAINABLE_PARAMETERS,
    NUM_NON_TRAINABLE_PARAMETERS,
)
from ray.rllib.utils.mindspore_utils import convert_to_mindspore_tensor, copy_mindspore_tensors
from ray.rllib.utils.typing import (
    ModuleID,
    Optimizer,
    Param,
    ParamDict,
    ShouldModuleBeUpdatedFn,
    StateDict,
    TensorType,
)

mindspore, nn = try_import_mindspore()

# if mindspore:
    # from ray.air._internal.mindspore_utils import get_devices


logger = logging.getLogger(__name__)


class MindSporeLearner(Learner):

    framework: str = "mindspore"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Will be set during build.
        self._device = None

        # Whether to compile the RL Module of this learner. This implies that the.
        # forward_train method of the RL Module will be compiled. Further more,
        # other forward methods of the RL Module will be compiled on demand.
        # This is assumed to not happen, since other forwrad methods are not expected
        # to be used during training.
        self._mindspore_compile_forward_train = False
        self._mindspore_compile_cfg = None
        # Whether to compile the `_uncompiled_update` method of this learner. This
        # implies that everything within `_uncompiled_update` will be compiled,
        # not only the forward_train method of the RL Module.
        # Note that this is experimental.
        # Note that this requires recompiling the forward methods once we add/remove
        # RL Modules.
        self._mindspore_compile_complete_update = False
        if self.config.mindspore_compile_learner:
            if (
                self.config.mindspore_compile_learner_what_to_compile
                == MindSporeCompileWhatToCompile.COMPLETE_UPDATE
            ):
                self._mindspore_compile_complete_update = True
                self._compiled_update_initialized = False
            else:
                self._mindspore_compile_forward_train = True

            self._mindspore_compile_cfg = MindSporeCompileConfig(
                mindspore_dynamo_backend=self.config.mindspore_compile_learner_dynamo_backend,
                mindspore_dynamo_mode=self.config.mindspore_compile_learner_dynamo_mode,
            )

        # Loss scalers for mixed precision training. Map optimizer names to
        # associated mindspore GradScaler objects.
        self._grad_scalers = None
        if self.config._mindspore_grad_scaler_class:
            self._grad_scalers = defaultdict(
                lambda: self.config._mindspore_grad_scaler_class()
            )
        self._lr_schedulers = {}
        self._lr_scheduler_classes = None
        if self.config._mindspore_lr_scheduler_classes:
            self._lr_scheduler_classes = self.config._mindspore_lr_scheduler_classes

    @OverrideToImplementCustomLogic
    @override(Learner)
    def configure_optimizers_for_module(
        self,
        module_id: ModuleID,
        config: "AlgorithmConfig" = None,
    ) -> None:
        module = self._module[module_id]

        # For this default implementation, the learning rate is handled by the
        # attached lr Scheduler (controlled by self.config.lr, which can be a
        # fixed value or a schedule setting).
        params = list(module.get_parameters())
        optimizer = mindspore.nn.Adam(params)

        # Register the created optimizer (under the default optimizer name).
        self.register_optimizer(
            module_id=module_id,
            optimizer=optimizer,
            params=params,
            lr_or_lr_schedule=config.lr,
        )

    def _uncompiled_update(
        self,
        batch: Dict,
        **kwargs,
    ):
        """Performs a single update given a batch of data."""
        # Activate tensor-mode on our MetricsLogger.
        self.metrics.activate_tensor_mode()

        fwd_out = self.module.forward_train(batch)
        loss_per_module = self.compute_losses(fwd_out=fwd_out, batch=batch)

        gradients = self.compute_gradients(loss_per_module)
        postprocessed_gradients = self.postprocess_gradients(gradients)
        self.apply_gradients(postprocessed_gradients)

        # Deactivate tensor-mode on our MetricsLogger and collect the (tensor)
        # results.
        collected_tensor_metrics = self.metrics.deactivate_tensor_mode()

        return fwd_out, loss_per_module, collected_tensor_metrics

    @override(Learner)
    def compute_gradients(
        self, loss_per_module: Dict[ModuleID, TensorType], **kwargs
    ) -> ParamDict:
        for optim in self._optimizer_parameters:
            # `set_to_none=True` is a faster way to zero out the gradients.
            optim.zero_grad(set_to_none=True)

        if self._grad_scalers is not None:
            total_loss = sum(
                self._grad_scalers[mid].scale(loss)
                for mid, loss in loss_per_module.items()
            )
        else:
            total_loss = sum(loss_per_module.values())

        total_loss.backward()
        grads = {pid: p.grad for pid, p in self._params.items()}

        return grads

    @override(Learner)
    def apply_gradients(self, gradients_dict: ParamDict) -> None:
        # Set the gradient of the parameters.
        for pid, grad in gradients_dict.items():
            # If updates should not be skipped turn `nan` and `inf` gradients to zero.
            if (
                not mindspore.isfinite(grad).all()
                and not self.config.mindspore_skip_nan_gradients
            ):
                # Warn the user about `nan` gradients.
                logger.warning(f"Gradients {pid} contain `nan/inf` values.")
                # If updates should be skipped, do not step the optimizer and return.
                if not self.config.mindspore_skip_nan_gradients:
                    logger.warning(
                        "Setting `nan/inf` gradients to zero. If updates with "
                        "`nan/inf` gradients should not be set to zero and instead "
                        "the update be skipped entirely set `mindspore_skip_nan_gradients` "
                        "to `True`."
                    )
                # If necessary turn `nan` gradients to zero. Note this can corrupt the
                # internal state of the optimizer, if many `nan` gradients occur.
                self._params[pid].grad = mindspore.nan_to_num(grad)
            # Otherwise, use the gradient as is.
            else:
                self._params[pid].grad = grad

        # For each optimizer call its step function.
        for module_id, optimizer_names in self._module_optimizers.items():
            for optimizer_name in optimizer_names:
                optim = self.get_optimizer(module_id, optimizer_name)
                # If we have learning rate schedulers for a module add them, if
                # necessary.
                if self._lr_scheduler_classes is not None:
                    if module_id not in self._lr_schedulers:
                        # Set for each module and optimizer a scheduler.
                        self._lr_schedulers[module_id] = {optimizer_name: []}
                        # If the classes are in a dictionary each module might have
                        # a different set of schedulers.
                        if isinstance(self._lr_scheduler_classes, dict):
                            scheduler_classes = self._lr_scheduler_classes[module_id]
                        # Else, each module has the same learning rate schedulers.
                        else:
                            scheduler_classes = self._lr_scheduler_classes
                        # Initialize and add the schedulers.
                        for scheduler_class in scheduler_classes:
                            self._lr_schedulers[module_id][optimizer_name].append(
                                scheduler_class(optim)
                            )

                # Step through the scaler (unscales gradients, if applicable).
                if self._grad_scalers is not None:
                    scaler = self._grad_scalers[module_id]
                    scaler.step(optim)
                    self.metrics.log_value(
                        (module_id, "_mindspore_grad_scaler_current_scale"),
                        scaler.get_scale(),
                        window=1,  # snapshot in time, no EMA/mean.
                    )
                    # Update the scaler.
                    scaler.update()
                # `step` the optimizer (default), but only if all gradients are finite.
                elif all(
                    param.grad is None or mindspore.isfinite(param.grad).all()
                    for group in optim.param_groups
                    for param in group["params"]
                ):
                    optim.step()
                # If gradients are not all finite warn the user that the update will be
                # skipped.
                elif not all(
                    mindspore.isfinite(param.grad).all()
                    for group in optim.param_groups
                    for param in group["params"]
                ):
                    logger.warning(
                        "Skipping this update. If updates with `nan/inf` gradients "
                        "should not be skipped entirely and instead `nan/inf` "
                        "gradients set to `zero` set `mindspore_skip_nan_gradients` to "
                        "`False`."
                    )

                    # If the module uses learning rate schedulers, step them here.
                    if module_id in self._lr_schedulers:
                        for scheduler in self._lr_schedulers[module_id][optimizer_name]:
                            scheduler.step()

                    # If the module uses learning rate schedulers, step them here.
                    if module_id in self._lr_schedulers:
                        for scheduler in self._lr_schedulers[module_id][optimizer_name]:
                            scheduler.step()

    @override(Learner)
    def _get_optimizer_state(self) -> StateDict:
        return {
            name: copy_mindspore_tensors(optim.state_dict(), device="cpu")
            for name, optim in self._named_optimizers.items()
        }

    @override(Learner)
    def _set_optimizer_state(self, state: StateDict) -> None:
        for name, state_dict in state.items():
            if name not in self._named_optimizers:
                raise ValueError(
                    f"Optimizer {name} in `state` is not known."
                    f"Known optimizers are {self._named_optimizers.keys()}"
                )
            self._named_optimizers[name].load_state_dict(
                copy_mindspore_tensors(state_dict, device=self._device)
            )

    @override(Learner)
    def get_param_ref(self, param: Param) -> Hashable:
        return param

    @override(Learner)
    def get_parameters(self, module: RLModule) -> Sequence[Param]:
        return list(module.parameters())

    @override(Learner)
    def _convert_batch_type(self, batch: MultiAgentBatch) -> MultiAgentBatch:
        batch = convert_to_mindspore_tensor(batch.policy_batches, device=self._device)
        # TODO (sven): This computation of `env_steps` is not accurate!
        length = max(len(b) for b in batch.values())
        batch = MultiAgentBatch(batch, env_steps=length)
        return batch

    @override(Learner)
    def add_module(
        self,
        *,
        module_id: ModuleID,
        # TODO (sven): Rename to `rl_module_spec`.
        module_spec: RLModuleSpec,
        config_overrides: Optional[Dict] = None,
        new_should_module_be_updated: Optional[ShouldModuleBeUpdatedFn] = None,
    ) -> MultiRLModuleSpec:
        # Call super's add_module method.
        marl_spec = super().add_module(
            module_id=module_id,
            module_spec=module_spec,
            config_overrides=config_overrides,
            new_should_module_be_updated=new_should_module_be_updated,
        )

        # we need to ddpify the module that was just added to the pool
        module = self._module[module_id]

        if self._mindspore_compile_forward_train:
            module.compile(self._mindspore_compile_cfg)
        elif self._mindspore_compile_complete_update:
            # When compiling the update, we need to reset and recompile
            # _uncompiled_update every time we add/remove a module anew.
            mindspore._dynamo.reset()
            self._compiled_update_initialized = False
            self._possibly_compiled_update = mindspore.compile(
                self._uncompiled_update,
                backend=self._mindspore_compile_cfg.mindspore_dynamo_backend,
                mode=self._mindspore_compile_cfg.mindspore_dynamo_mode,
                **self._mindspore_compile_cfg.kwargs,
            )

        if isinstance(module, MindSporeRLModule):
            self._module[module_id].to(self._device)
            if self.distributed:
                if (
                    self._mindspore_compile_complete_update
                    or self._mindspore_compile_forward_train
                ):
                    raise ValueError(
                        "Using mindspore distributed and mindspore compile "
                        "together tested for now. Please disable "
                        "mindspore compile."
                    )
                self._module.add_module(
                    module_id,
                    MindSporeDDPRLModule(module, **self.config.mindspore_ddp_kwargs),
                    override=True,
                )

        return marl_spec

    @override(Learner)
    def remove_module(self, module_id: ModuleID, **kwargs) -> MultiRLModuleSpec:
        marl_spec = super().remove_module(module_id, **kwargs)

        if self._mindspore_compile_complete_update:
            # When compiling the update, we need to reset and recompile
            # _uncompiled_update every time we add/remove a module anew.
            mindspore._dynamo.reset()
            self._compiled_update_initialized = False
            self._possibly_compiled_update = mindspore.compile(
                self._uncompiled_update,
                backend=self._mindspore_compile_cfg.mindspore_dynamo_backend,
                mode=self._mindspore_compile_cfg.mindspore_dynamo_mode,
                **self._mindspore_compile_cfg.kwargs,
            )

        return marl_spec

    @override(Learner)
    def build(self) -> None:
        """Builds the MindSporeLearner.

        This method is specific to MindSporeLearner. Before running super() it will
        initialze the device properly based on the `_use_gpu` and `_distributed`
        flags, so that `_make_module()` can place the created module on the correct
        device. After running super() it will wrap the module in a MindSporeDDPRLModule
        if `_distributed` is True.
        Note, in inherited classes it is advisable to call the parent's `build()`
        after setting up all variables because `configure_optimizer_for_module` is
        called in this `Learner.build()`.
        """
        # TODO (Kourosh): How do we handle model parallelism?
        # TODO (Kourosh): Instead of using _MindSporeAccelerator, we should use the public
        #  API in ray.train but allow for session to be None without any errors raised.
        if self._use_gpu:
            # get_devices() returns a list that contains the 0th device if
            # it is called from outside of a Ray Train session. Its necessary to give
            # the user the option to run on the gpu of their choice, so we enable that
            # option here via the local gpu id scaling config parameter.
            if self._distributed:
                # devices = get_devices()
                # assert len(devices) == 1, (
                #     "`get_devices()` should only return one cuda device, "
                #     f"but {devices} was returned instead."
                # )
                self._device = mindspore.hal.get_device_name(0)
            else:
                assert self._local_gpu_idx < mindspore.cuda.device_count(), (
                    f"local_gpu_idx {self._local_gpu_idx} is not a valid GPU id or is "
                    " not available."
                )
                # this is an index into the available cuda devices. For example if
                # os.environ["CUDA_VISIBLE_DEVICES"] = "1" then
                # mindspore.cuda.device_count() = 1 and mindspore.device(0) will actuall map to
                # the gpu with id 1 on the node.
                self._device = mindspore.hal.get_device_name(self._local_gpu_idx)
        else:
            self._device = mindspore.hal.get_device_name(0, device_target="CPU")
            
        super().build()

        if self._mindspore_compile_complete_update:
            mindspore._dynamo.reset()
            self._compiled_update_initialized = False
            self._possibly_compiled_update = mindspore.compile(
                self._uncompiled_update,
                backend=self._mindspore_compile_cfg.mindspore_dynamo_backend,
                mode=self._mindspore_compile_cfg.mindspore_dynamo_mode,
                **self._mindspore_compile_cfg.kwargs,
            )
        else:
            if self._mindspore_compile_forward_train:
                if isinstance(self._module, MindSporeRLModule):
                    self._module.compile(self._mindspore_compile_cfg)
                elif isinstance(self._module, MultiRLModule):
                    for module in self._module._rl_modules.values():
                        # Compile only MindSporeRLModules, e.g. we don't want to compile
                        # a RandomRLModule.
                        if isinstance(self._module, MindSporeRLModule):
                            module.compile(self._mindspore_compile_cfg)
                else:
                    raise ValueError(
                        "MindSpore compile is only supported for MindSporeRLModule and "
                        "MultiRLModule."
                    )

            self._possibly_compiled_update = self._uncompiled_update

        self._make_modules_ddp_if_necessary()

        # Log number of non-trainable and trainable parameters of our RLModule.
        num_trainable_params = {
            (mid, NUM_TRAINABLE_PARAMETERS): sum(
                p.numel() for p in rlm.get_parameters() if p.requires_grad
            )
            for mid, rlm in self.module._rl_modules.items()
            if isinstance(rlm, MindSporeRLModule)
        }
        num_non_trainable_params = {
            (mid, NUM_NON_TRAINABLE_PARAMETERS): sum(
                p.numel() for p in rlm.get_parameters() if not p.requires_grad
            )
            for mid, rlm in self.module._rl_modules.items()
            if isinstance(rlm, MindSporeRLModule)
        }
        self.metrics.log_dict(
            {
                **{
                    (ALL_MODULES, NUM_TRAINABLE_PARAMETERS): sum(
                        num_trainable_params.values()
                    ),
                    (ALL_MODULES, NUM_NON_TRAINABLE_PARAMETERS): sum(
                        num_non_trainable_params.values()
                    ),
                },
                **num_trainable_params,
                **num_non_trainable_params,
            }
        )

    @override(Learner)
    def _update(self, batch: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        # The first time we call _update after building the learner or
        # adding/removing models, we update with the uncompiled update method.
        # This makes it so that any variables that may be created during the first
        # update step are already there when compiling. More specifically,
        # this avoids errors that occur around using defaultdicts with
        # mindspore.compile().
        if (
            self._mindspore_compile_complete_update
            and not self._compiled_update_initialized
        ):
            self._compiled_update_initialized = True
            return self._uncompiled_update(batch)
        else:
            return self._possibly_compiled_update(batch)

    @OverrideToImplementCustomLogic
    def _make_modules_ddp_if_necessary(self) -> None:
        """Default logic for (maybe) making all Modules within self._module DDP."""

        # If the module is a MultiRLModule and nn.Cell we can simply assume
        # all the submodules are registered. Otherwise, we need to loop through
        # each submodule and move it to the correct device.
        # TODO (Kourosh): This can result in missing modules if the user does not
        #  register them in the MultiRLModule. We should find a better way to
        #  handle this.
        if self._distributed:
            # Single agent module: Convert to `MindSporeDDPRLModule`.
            if isinstance(self._module, MindSporeRLModule):
                self._module = MindSporeDDPRLModule(
                    self._module, **self.config.mindspore_ddp_kwargs
                )
            # Multi agent module: Convert each submodule to `MindSporeDDPRLModule`.
            else:
                assert isinstance(self._module, MultiRLModule)
                for key in self._module.keys():
                    sub_module = self._module[key]
                    if isinstance(sub_module, MindSporeRLModule):
                        # Wrap and override the module ID key in self._module.
                        self._module.add_module(
                            key,
                            MindSporeDDPRLModule(
                                sub_module, **self.config.mindspore_ddp_kwargs
                            ),
                            override=True,
                        )

    def _is_module_compatible_with_learner(self, module: RLModule) -> bool:
        return isinstance(module, nn.Cell)

    @override(Learner)
    def _check_registered_optimizer(
        self,
        optimizer: Optimizer,
        params: Sequence[Param],
    ) -> None:
        super()._check_registered_optimizer(optimizer, params)
        if not isinstance(optimizer, mindspore.nn.Optimizer):
            raise ValueError(
                f"The optimizer ({optimizer}) is not a mindspore.nn.Optimizer! "
                "Only use mindspore.nn.Optimizer subclasses for MindSporeLearner."
            )
        for param in params:
            if not isinstance(param, mindspore.Tensor):
                raise ValueError(
                    f"One of the parameters ({param}) in the registered optimizer "
                    "is not a mindspore.Tensor!"
                )

    @override(Learner)
    def _make_module(self) -> MultiRLModule:
        module = super()._make_module()
        self._map_module_to_device(module)
        return module

    def _map_module_to_device(self, module: MultiRLModule) -> None:
        """Moves the module to the correct device."""
        if isinstance(module, mindspore.nn.Cell):
            module.to(self._device)
        else:
           # for key in module.keys():
               # if isinstance(module[key], mindspore.nn.Cell):
                   # module[key].to(self._device)
           pass

    @override(Learner)
    def _get_tensor_variable(
        self, value, dtype=None, trainable=False
    ) -> "mindspore.Tensor":
        tensor = mindspore.tensor(
            value,
            requires_grad=trainable,
            device=self._device,
            dtype=(
                dtype
                or (
                    mindspore.float32
                    if isinstance(value, float)
                    else mindspore.int32
                    if isinstance(value, int)
                    else None
                )
            ),
        )
        return nn.Parameter(tensor) if trainable else tensor

    @staticmethod
    @override(Learner)
    def _get_optimizer_lr(optimizer: "mindspore.nn.Optimizer") -> float:
        for g in optimizer.get_parameters():
            return g["lr"]

    @staticmethod
    @override(Learner)
    def _set_optimizer_lr(optimizer: "mindspore.nn.Optimizer", lr: float) -> None:
        for g in optimizer.get_parameters():
          if g.name == 'lr':
            g = lr

    @staticmethod
    @override(Learner)
    def _get_clip_function() -> Callable:
        from ray.rllib.utils.mindspore_utils import clip_gradients

        return clip_gradients

    @staticmethod
    @override(Learner)
    def _get_global_norm_function() -> Callable:
        from ray.rllib.utils.mindspore_utils import compute_global_norm

        return compute_global_norm
