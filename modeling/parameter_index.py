from __future__ import annotations

from typing import TYPE_CHECKING

import transformers
from transformers import AutoModelForCausalLM

from modeling.deferred import DeferredTensor, defer_tensor_load
from modeling.util import get_model_keys

if TYPE_CHECKING:
    from modeling.modeling import Model


class index_tensors:
    def __init__(self, index: ParameterIndex) -> None:
        self.param_index = index
        self._unpatched = transformers.modeling_utils._load_state_dict_into_meta_model

    def __enter__(self) -> None:
        def _cache_tensors(model, state_dict, *args, **kwargs):
            for tensor_name, deferred_tensor in state_dict.items():
                self.param_index.deferred_tensors[tensor_name] = deferred_tensor
            return [], None, None

        # `_load_state_dict_into_meta_model` chosen for easy access to state_dict,
        # and I'm already familiar with it. There may be a higher-level function
        # called earlier in the callstack which could potentially reduce memory overhead.
        transformers.modeling_utils._load_state_dict_into_meta_model = _cache_tensors

        # Don't confuse the user with progress bars for fake models!
        transformers.utils.logging.disable_progress_bar()

    def __exit__(self, *args) -> None:
        transformers.modeling_utils._load_state_dict_into_meta_model = self._unpatched
        transformers.utils.logging.enable_progress_bar()


class ParameterIndex:
    def __init__(self, path: str):
        self.deferred_tensors: dict[str, DeferredTensor] = {}

        device_map = {k: "meta" for k in get_model_keys(path)}

        # This context manager writes DeferredTensors to `self.deferred_tensors`
        # instead of actually loading weights.
        with defer_tensor_load(), index_tensors(self):
            try:
                # Using the meta device saves a significant amount of memory over
                # CPU, despite no weights loading (in theory). Patching in a
                # better place level function might fix that but using the meta
                # device is an easier fix :P
                print(f"[index] Indexing '{path}'...")
                AutoModelForCausalLM.from_pretrained(path, device_map=device_map)
            except ValueError as e:
                # Actual model load fails because weights aren't loaded (which is what we want)
                if "is on the meta device, we need a `value`" not in str(e):
                    raise e

    def __getitem__(self, key: str) -> DeferredTensor:
        return self.deferred_tensors[key]
