import logging
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
import mindspore

from ray.air._internal.mindspore_utils import convert_ndarray_batch_to_mindspore_tensor_batch
from ray.train._internal.dl_predictor import DLPredictor
from ray.train.predictor import DataBatchType
from ray.train.mindspore import MindSporeCheckpoint
from ray.util import log_once
from ray.util.annotations import DeveloperAPI, PublicAPI

if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class MindSporePredictor(DLPredictor):
    """A predictor for MindSpore models.

    Args:
        model: The mindspore module to use for predictions.
        preprocessor: A preprocessor used to transform data batches prior
            to prediction.
        use_gpu: If set, the model will be moved to GPU on instantiation and
            prediction happens on GPU.
    """

    def __init__(
        self,
        model: mindspore.nn.Cell,
        preprocessor: Optional["Preprocessor"] = None,
        use_gpu: bool = False,
    ):
        self.model = model
        self.model.eval()
        self.use_gpu = use_gpu

        # if use_gpu:
        #     # TODO (jiaodong): #26249 Use multiple GPU devices with sharded input
        #     self.device = mindspore.device("cuda")
        # else:
        #     self.device = mindspore.device("cpu")
        #
        # # Ensure input tensor and model live on the same device
        # self.model.to(self.device)
        #
        # if (
        #     not use_gpu
        #     and mindspore.cuda.device_count() > 0
        #     and log_once("mindspore_predictor_not_using_gpu")
        # ):
        #     logger.warning(
        #         "You have `use_gpu` as False but there are "
        #         f"{mindspore.cuda.device_count()} GPUs detected on host where "
        #         "prediction will only use CPU. Please consider explicitly "
        #         "setting `MindSporePredictor(use_gpu=True)` or "
        #         "`batch_predictor.predict(ds, num_gpus_per_worker=1)` to "
        #         "enable GPU prediction."
        #     )
        if use_gpu:
            # 使用多个GPU设备，输入分片
            self.device = mindspore_device("cuda")
        else:
            # 使用CPU
            self.device = mindspore_device("cpu")

        # 确保输入张量和模型位于同一设备
        self.model.to(self.device)

        if not use_gpu and cuda.device_count() > 0 and log_once("mindspore_predictor_not_using_gpu"):
            logger.warning(
                "You have `use_gpu` as False but there are "
                f"{cuda.device_count()} GPUs detected on host where "
                "prediction will only use CPU. Please consider explicitly "
                "setting `MindSporePredictor(use_gpu=True)` or "
                "`batch_predictor.predict(ds, num_gpus_per_worker=1)` to "
                "enable GPU prediction."
            )
        super().__init__(preprocessor)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(model={self.model!r}, "
            f"preprocessor={self._preprocessor!r}, use_gpu={self.use_gpu!r})"
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: MindSporeCheckpoint,
        model: Optional[mindspore.nn.Cell] = None,
        use_gpu: bool = False,
    ) -> "MindSporePredictor":
        """Instantiate the predictor from a MindSporeCheckpoint.

        Args:
            checkpoint: The checkpoint to load the model and preprocessor from.
            model: If the checkpoint contains a model state dict, and not
                the model itself, then the state dict will be loaded to this
                ``model``. If the checkpoint already contains the model itself,
                this model argument will be discarded.
            use_gpu: If set, the model will be moved to GPU on instantiation and
                prediction happens on GPU.
        """
        model = checkpoint.get_model(model)
        preprocessor = checkpoint.get_preprocessor()
        return cls(model=model, preprocessor=preprocessor, use_gpu=use_gpu)

    @DeveloperAPI
    def call_model(
        self, inputs: Union[mindspore.Tensor, Dict[str, mindspore.Tensor]]
    ) -> Union[mindspore.Tensor, Dict[str, mindspore.Tensor]]:
        """Runs inference on a single batch of tensor data.

        This method is called by `MindSporePredictor.predict` after converting the
        original data batch to mindspore tensors.

        Override this method to add custom logic for processing the model input or
        output.

        Args:
            inputs: A batch of data to predict on, represented as either a single
                MindSpore tensor or for multi-input models, a dictionary of tensors.

        Returns:
            The model outputs, either as a single tensor or a dictionary of tensors.

        Example:

            .. testcode::

                import numpy as np
                import mindspore
                from ray.train.mindspore import MindSporePredictor

                # List outputs are not supported by default MindSporePredictor.
                # So let's define a custom MindSporePredictor and override call_model
                class MyModel(mindspore.nn.Cell):
                    def forward(self, input_tensor):
                        return [input_tensor, input_tensor]

                # Use a custom predictor to format model output as a dict.
                class CustomPredictor(MindSporePredictor):
                    def call_model(self, inputs):
                        model_output = super().call_model(inputs)
                        return {
                            str(i): model_output[i] for i in range(len(model_output))
                        }

                # create our data batch
                data_batch = np.array([1, 2])
                # create custom predictor and predict
                predictor = CustomPredictor(model=MyModel())
                predictions = predictor.predict(data_batch)
                print(f"Predictions: {predictions.get('0')}, {predictions.get('1')}")

            .. testoutput::

                Predictions: [1 2], [1 2]

        """
        # with mindspore.no_grad():
        #     output = self.model(inputs)
        # return output

        is_training = self.model.training

        # 将模型设置为非训练模式，即评估模式，这样在前向传播时不会计算梯度
        self.model.set_train(False)

        try:
            # 执行模型的前向传播，不计算梯度
            output = self.model(inputs)
        finally:
            # 将模型设置回原来的状态
            self.model.set_train(is_training)

        # 返回输出结果
        return output

    def predict(
        self,
        data: DataBatchType,
        dtype: Optional[Union[mindspore.dtype, Dict[str, mindspore.dtype]]] = None,
    ) -> DataBatchType:
        """Run inference on data batch.

        If the provided data is a single array or a dataframe/table with a single
        column, it will be converted into a single PyMindSpore tensor before being
        inputted to the model.

        If the provided data is a multi-column table or a dict of numpy arrays,
        it will be converted into a dict of tensors before being inputted to the
        model. This is useful for multi-modal inputs (for example your model accepts
        both image and text).

        Args:
            data: A batch of input data of ``DataBatchType``.
            dtype: The dtypes to use for the tensors. Either a single dtype for all
                tensors or a mapping from column name to dtype.

        Returns:
            DataBatchType: Prediction result. The return type will be the same as the
                input type.

        Example:

            .. testcode::

                    import numpy as np
                    import pandas as pd
                    import mindspore
                    import ray
                    from ray.train.mindspore import MindSporePredictor

                    # Define a custom PyMindSpore module
                    class CustomModule(mindspore.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.linear1 = mindspore.nn.Linear(1, 1)
                            self.linear2 = mindspore.nn.Linear(1, 1)

                        def forward(self, input_dict: dict):
                            out1 = self.linear1(input_dict["A"].unsqueeze(1))
                            out2 = self.linear2(input_dict["B"].unsqueeze(1))
                            return out1 + out2

                    # Set manul seed so we get consistent output
                    mindspore.manual_seed(42)

                    # Use Standard PyMindSpore model
                    model = mindspore.nn.Linear(2, 1)
                    predictor = MindSporePredictor(model=model)
                    # Define our data
                    data = np.array([[1, 2], [3, 4]])
                    predictions = predictor.predict(data, dtype=mindspore.float)
                    print(f"Standard model predictions: {predictions}")
                    print("---")

                    # Use Custom PyMindSpore model with MindSporePredictor
                    predictor = MindSporePredictor(model=CustomModule())
                    # Define our data and predict Customer model with MindSporePredictor
                    data = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
                    predictions = predictor.predict(data, dtype=mindspore.float)
                    print(f"Custom model predictions: {predictions}")

            .. testoutput::

                Standard model predictions: {'predictions': array([[1.5487633],
                       [3.8037925]], dtype=float32)}
                ---
                Custom model predictions:     predictions
                0  [0.61623406]
                1    [2.857038]
        """
        return super(MindSporePredictor, self).predict(data=data, dtype=dtype)

    def _arrays_to_tensors(
        self,
        numpy_arrays: Union[np.ndarray, Dict[str, np.ndarray]],
        dtype: Optional[Union[mindspore.dtype, Dict[str, mindspore.dtype]]],
    ) -> Union[mindspore.Tensor, Dict[str, mindspore.Tensor]]:
        return convert_ndarray_batch_to_mindspore_tensor_batch(
            numpy_arrays,
            dtypes=dtype,
            device=self.device,
        )

    def _tensor_to_array(self, tensor: mindspore.Tensor) -> np.ndarray:
        if not isinstance(tensor, mindspore.Tensor):
            raise ValueError(
                "Expected the model to return either a mindspore.Tensor or a "
                f"dict of mindspore.Tensor, but got {type(tensor)} instead. "
                f"To support models with different output types, subclass "
                f"MindSporePredictor and override the `call_model` method to "
                f"process the output into either mindspore.Tensor or Dict["
                f"str, mindspore.Tensor]."
            )
        return tensor.cpu().detach().numpy()
