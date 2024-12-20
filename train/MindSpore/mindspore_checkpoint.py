import os
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import mindspore

from ray.air._internal.mindspore_utils import (
    consume_prefix_in_state_dict_if_present_not_in_place,
    load_mindspore_model,
)
from ray.train._internal.framework_checkpoint import FrameworkCheckpoint
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor

ENCODED_DATA_KEY = "mindspore_encoded_data"


@PublicAPI(stability="beta")
class MindSporeCheckpoint(FrameworkCheckpoint):
    """A :class:`~ray.train.Checkpoint` with MindSpore-specific functionality."""

    MODEL_FILENAME = "model.pt"

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, Any],
        *,
        preprocessor: Optional["Preprocessor"] = None,
    ) -> "MindSporeCheckpoint":
        """Create a :class:`~ray.train.Checkpoint` that stores a model state dictionary.

        .. tip::

            This is the recommended method for creating
            :class:`MindSporeCheckpoints<MindSporeCheckpoint>`.

        Args:
            state_dict: The model state dictionary to store in the checkpoint.
            preprocessor: A fitted preprocessor to be applied before inference.

        Returns:
            A :class:`MindSporeCheckpoint` containing the specified state dictionary.

        Examples:

            .. testcode::

                import mindspore
                import mindspore.nn as nn
                from ray.train.mindspore import MindSporeCheckpoint

                # Set manual seed
                mindspore.manual_seed(42)

                # Function to create a NN model
                def create_model() -> nn.Module:
                    model = nn.Sequential(nn.Linear(1, 10),
                            nn.ReLU(),
                            nn.Linear(10,1))
                    return model

                # Create a MindSporeCheckpoint from our model's state_dict
                model = create_model()
                checkpoint = MindSporeCheckpoint.from_state_dict(model.state_dict())

                # Now load the model from the MindSporeCheckpoint by providing the
                # model architecture
                model_from_chkpt = checkpoint.get_model(create_model())

                # Assert they have the same state dict
                assert str(model.state_dict()) == str(model_from_chkpt.state_dict())
                print("worked")

            .. testoutput::
                :hide:

                ...
        """
        tempdir = tempfile.mkdtemp()

        model_path = Path(tempdir, cls.MODEL_FILENAME).as_posix()
        stripped_state_dict = consume_prefix_in_state_dict_if_present_not_in_place(
            state_dict, "module."
        )
        mindspore.save(stripped_state_dict, model_path)

        checkpoint = cls.from_directory(tempdir)
        if preprocessor:
            checkpoint.set_preprocessor(preprocessor)
        return checkpoint

    @classmethod
    def from_model(
        cls,
        model: mindspore.nn.Cell,
        *,
        preprocessor: Optional["Preprocessor"] = None,
    ) -> "MindSporeCheckpoint":
        """Create a :class:`~ray.train.Checkpoint` that stores a MindSpore model.

        .. note::

            PyMindSpore recommends storing state dictionaries. To create a
            :class:`MindSporeCheckpoint` from a state dictionary, call
            :meth:`~ray.train.mindspore.MindSporeCheckpoint.from_state_dict`. To learn more
            about state dictionaries, read
            `Saving and Loading Models <https://pymindspore.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict>`_. # noqa: E501

        Args:
            model: The MindSpore model to store in the checkpoint.
            preprocessor: A fitted preprocessor to be applied before inference.

        Returns:
            A :class:`MindSporeCheckpoint` containing the specified model.

        Examples:

            .. testcode::

                from ray.train.mindspore import MindSporeCheckpoint
                import mindspore

                # Create model identity and send a random tensor to it
                model = mindspore.nn.Identity()
                input = mindspore.randn(2, 2)
                output = model(input)

                # Create a checkpoint
                checkpoint = MindSporeCheckpoint.from_model(model)
                print(checkpoint)

            .. testoutput::
                :hide:

                ...
        """
        tempdir = tempfile.mkdtemp()

        model_path = Path(tempdir, cls.MODEL_FILENAME).as_posix()
        mindspore.save(model, model_path)

        checkpoint = cls.from_directory(tempdir)
        if preprocessor:
            checkpoint.set_preprocessor(preprocessor)
        return checkpoint

    def get_model(self, model: Optional[mindspore.nn.Cell] = None) -> mindspore.nn.Cell:
        """Retrieve the model stored in this checkpoint.

        Args:
            model: If the checkpoint contains a model state dict, and not
                the model itself, then the state dict will be loaded to this
                ``model``. Otherwise, the model will be discarded.
        """
        with self.as_directory() as tempdir:
            model_path = Path(tempdir, self.MODEL_FILENAME).as_posix()
            if not os.path.exists(model_path):
                raise RuntimeError(
                    "`model.pt` not found within this checkpoint. Make sure you "
                    "created this `MindSporeCheckpoint` from one of its public "
                    "constructors (`from_state_dict` or `from_model`)."
                )
            model_or_state_dict = mindspore.load(model_path, map_location="cpu")

        if isinstance(model_or_state_dict, mindspore.nn.Cell):
            if model:
                warnings.warn(
                    "MindSporeCheckpoint already contains all information needed. "
                    "Discarding provided `model` argument. If you are using "
                    "MindSporePredictor directly, you should do "
                    "`MindSporePredictor.from_checkpoint(checkpoint)` by removing kwargs "
                    "`model=`."
                )
        model = load_mindspore_model(
            saved_model=model_or_state_dict, model_definition=model
        )
        return model
