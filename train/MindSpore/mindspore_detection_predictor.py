import collections
from typing import Dict, List, Optional, Union

import numpy as np
import mindspore

from ray.train._internal.dl_predictor import TensorDtype
from ray.train.MindSpore.mindspore_predictor import MindSporePredictor
from ray.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class MindSporeDetectionPredictor(MindSporePredictor):
    """A predictor for MindSporeVision detection models.

    Unlike other MindSpore models, instance segmentation models return
    `List[Dict[str, Tensor]]`. This predictor extends :class:`MindSporePredictor` to support
    the non-standard outputs.

    To learn more about instance segmentation models, read
    `Instance segmentation models <https://pymindspore.org/vision/main/auto_examples/plot_visualization_utils.html#instance-seg-output>`_.

    Example:

        .. testcode::

            import numpy as np
            from mindsporevision import models

            from ray.train.mindspore import MindSporeDetectionPredictor

            model = models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)

            predictor = MindSporeDetectionPredictor(model=model)
            predictions = predictor.predict(np.zeros((4, 3, 32, 32), dtype=np.float32))

            print(predictions.keys())

        .. testoutput::

            dict_keys(['pred_boxes', 'pred_labels', 'pred_scores'])

    """  # noqa: E501

    def _predict_numpy(
        self,
        data: Union[np.ndarray, Dict[str, np.ndarray]],
        dtype: Optional[Union[TensorDtype, Dict[str, TensorDtype]]],
    ) -> Dict[str, np.ndarray]:
        if isinstance(data, dict) and len(data) != 1:
            raise ValueError(
                f"""Expected input to contain one key, but got {len(data)} instead."""
            )

        if dtype is not None and not isinstance(dtype, mindspore.dtype):
            raise ValueError(
                "Expected `dtype` to be a `mindspore.dtype`, but got a "
                f"{type(dtype).__name__} instead."
            )

        if isinstance(data, dict):
            images = next(iter(data.values()))
        else:
            images = data

        inputs = [
            mindspore.ops.is_tensor(image, dtype=dtype).to(self.device) for image in images
        ]
        outputs = self.call_model(inputs)
        outputs = _convert_outputs_to_batch(outputs)
        outputs = {"pred_" + key: value for key, value in outputs.items()}

        return outputs


def _convert_outputs_to_batch(
    outputs: List[Dict[str, mindspore.Tensor]],
) -> Dict[str, List[mindspore.Tensor]]:
    """Batch detection model outputs.

    MindSporeVision detection models return `List[Dict[Tensor]]`. Each `Dict` contain
    'boxes', 'labels, and 'scores'.

    This function batches values and returns a `Dict[str, List[Tensor]]`.
    """  # noqa: E501
    batch = collections.defaultdict(list)
    for output in outputs:
        for key, value in output.items():
            batch[key].append(value.cpu().detach())
    return batch
