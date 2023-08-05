# This script merges models with a goal of using as little memory as possible
# It's basically an isolated version of KoboldAI's deferred loader, and some new
# code from my `live-merge` branch
from __future__ import annotations

import argparse
import functools
import os
import pickle
import zipfile
from dataclasses import dataclass
from io import BufferedReader
from typing import Any, List, Optional

import accelerate
import safetensors
import safetensors.torch
import torch
import transformers
from torch.storage import UntypedStorage
from tqdm import tqdm
from transformers import AutoModelForCausalLM

DEVICE_MAP = "cpu"
MODEL_PRECISION = torch.float16
HF_KWARGS = {
    "torch_dtype": MODEL_PRECISION,
}

load_progress = None


def merge_tensors(targets: List[MergeTargetTensor], param_name: str) -> torch.Tensor:
    # There's probably a pretty torch function for this
    out = targets[0].tensor * targets[0].weight
    for target in targets[1:]:
        if target.tensor.shape != out.shape:
            print(f"[!] Shape mismatch on '{param_name}")
            continue
        out += target.tensor * target.weight
    return out


@dataclass
class MergeTargetTensor:
    """Structure for holding tensors with their weight/ratio/proportion/etc in
    the merge."""

    tensor: torch.Tensor
    weight: float


@dataclass
class CheckpointChunk:
    file_name: str
    key: str
    handle: BufferedReader


class CheckpointChunkCache:
    """Storage for common checkpoint weight files to speed up loading. In order
    for this to be effective at all, weights must be loaded in ascending order
    of (key, seek_offset).
    """

    # There is considerable room for improvement here; we could peek into the
    # state dict and preload the N most frequent weight files or something, but
    # this first implementation is on par with the speed of whatever the
    # previous callback did.

    chunks = {}
    file_handles = {}

    @classmethod
    def clear(cls, cleanup: bool = False, key: Optional[str] = None) -> None:
        if cleanup:
            for v in cls.file_handles.values():
                v.close()
            cls.file_handles.clear()

            for chunk in cls.chunks.values():
                if chunk.handle:
                    chunk.handle.close()
            cls.chunks.clear()

        if key and cls.chunks[key].handle:
            cls.chunks[key].handle.close()
            del cls.chunks[key]


class DeferredTensor:
    pass


class TorchDeferredTensor(DeferredTensor):
    def __init__(
        self,
        storage_type,
        key: str,
        location: str,
    ):
        self.storage_type = storage_type
        self.key = key
        self.location = location
        self.file_name = None

    def materialize(self, device=None) -> torch.Tensor:
        checkpoint = CheckpointChunkCache.file_handles[self.file_name]
        filename = os.path.basename(os.path.normpath(self.file_name)).split(".")[0]
        active_chunk = CheckpointChunkCache.chunks.get(self.file_name)

        # Often we are using the same weight file to store multiple tensors, so
        # let's cache the file handle to maintain a seek position and other
        # fast stuff.
        if (
            not active_chunk
            or active_chunk.file_name != filename
            or active_chunk.key != self.key
        ):
            # Cache miss. Assuming weights are loaded in order of
            # (key, seek_offset), this means we need to invalidate the cache.
            if active_chunk:
                CheckpointChunkCache.clear(key=self.file_name)

            ziproot = checkpoint.namelist()[0].split("/")[0]
            CheckpointChunkCache.chunks[self.file_name] = CheckpointChunk(
                filename, self.key, checkpoint.open(f"{ziproot}/data/{self.key}", "r")
            )
            active_chunk = CheckpointChunkCache.chunks[self.file_name]
            # print("!", end="", flush=True)
        else:
            # Cache hit. Hip hip hooray! :^)
            # print(".", end="", flush=True)
            pass

        size = functools.reduce(lambda x, y: x * y, self.shape, 1)
        dtype = self.dtype
        nbytes = (
            size
            if dtype is torch.bool
            else size
            * (
                (torch.finfo if self.dtype.is_floating_point else torch.iinfo)(
                    self.dtype
                ).bits
                >> 3
            )
        )

        assert isinstance(checkpoint, zipfile.ZipFile)

        active_chunk.handle.seek(self.seek_offset, os.SEEK_SET)
        storage = UntypedStorage.from_buffer(
            active_chunk.handle.read(nbytes), "little", dtype=self.dtype
        )

        storage = torch.serialization._get_restore_location(device)(
            storage, self.location
        )
        tensor = torch.tensor([], dtype=self.dtype, device=storage.device)
        tensor.set_(storage, 0, self.shape, self.stride)
        tensor.requires_grad = False
        if load_progress is not None:
            load_progress.update(nbytes)
        return tensor


class SafetensorsDeferredTensor(DeferredTensor):
    def __init__(self, checkpoint_file: str, key: str, location: str):
        self.checkpoint_file = checkpoint_file
        self.key = key
        self.location = location

    def materialize(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # keeping the file handle open and reading all at once drags everything
        # into ram, just open and close, as hacky as it is
        with safetensors.safe_open(
            self.checkpoint_file, framework="pt", device=self.location
        ) as f:
            return f.get_tensor(self.key)


class defer_tensor_load:
    class DeferredUnpickler(pickle.Unpickler):
        def persistent_load(self, saved_id):
            assert isinstance(saved_id, tuple)
            typename = saved_id[0]
            assert (
                typename == "storage"
            ), f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
            storage_type, key, location, _ = saved_id[1:]
            return TorchDeferredTensor(storage_type, key, location)

    def _patched_rebuild_tensor(
        deferred_tensor: DeferredTensor,
        storage_offset: int,
        shape: torch.Size,
        stride: torch.Size,
    ):
        deferred_tensor.shape = shape
        deferred_tensor.stride = stride
        dtype = deferred_tensor.storage_type.dtype
        if not isinstance(dtype, torch.dtype):
            dtype = deferred_tensor.storage_type(0).dtype
        deferred_tensor.dtype = dtype
        deferred_tensor.seek_offset = (
            storage_offset
            if dtype is torch.bool
            else storage_offset
            * (
                (torch.finfo if dtype.is_floating_point else torch.iinfo)(dtype).bits
                >> 3
            )
        )
        return deferred_tensor

    def _patched_load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
        model_dict = defer_tensor_load._patched_load._unpatched(
            f=f,
            map_location=map_location,
            pickle_module=pickle_module,
            **pickle_load_args,
        )

        if f not in CheckpointChunkCache.file_handles:
            CheckpointChunkCache.file_handles[f] = zipfile.ZipFile(f, "r")

        for deferred_tensor in model_dict.values():
            deferred_tensor.file_name = f

        return model_dict

    def _patched_safetensors_load(path: str) -> dict:
        tensors = {}

        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = None

        for key in tensors:
            tensors[key] = SafetensorsDeferredTensor(
                checkpoint_file=path, key=key, location="cpu"
            )
        return tensors

    def __init__(self) -> None:
        self._unpatched_pickler = pickle.Unpickler
        self._unpatched_torch_rebuild = torch._utils._rebuild_tensor
        self._unpatched_safetensors_load = safetensors.torch.load_file
        # HACK: Yeaaaaaaaaaah this isn't great
        defer_tensor_load._patched_load._unpatched = torch.load

    def __enter__(self) -> None:
        pickle.Unpickler = defer_tensor_load.DeferredUnpickler
        torch._utils._rebuild_tensor = defer_tensor_load._patched_rebuild_tensor
        torch.load = defer_tensor_load._patched_load

        transformers.modeling_utils.safe_load_file = (
            defer_tensor_load._patched_safetensors_load
        )
        safetensors.torch.load_file = defer_tensor_load._patched_safetensors_load

    def __exit__(self, *args) -> None:
        pickle.Unpickler = self._unpatched_pickler
        torch._utils._rebuild_tensor = self._unpatched_torch_rebuild
        torch.load = defer_tensor_load._patched_load._unpatched

        transformers.modeling_utils.safe_load_file = self._unpatched_safetensors_load
        safetensors.torch.load_file = self._unpatched_safetensors_load


class merge_with_secondary_models:
    class PatchedStateDict(dict):
        """Passed into _load_state_dict_into_meta_model as `state_dict` to avoid
        having to copy that whole super duper big function into here to patch a
        few lines."""

        def __init__(
            self, state_dict: dict, primary_model: Model, secondary_models: List[Model]
        ) -> None:
            self._state_dict = state_dict
            self._primary_model = primary_model
            self._secondary_models = secondary_models

        def items(self) -> Any:
            for param_name, param in sorted(
                self._state_dict.items(),
                # Make effecient use of checkpoint chunk cache by ordering by
                # position on disk
                key=lambda x: (x[1].key, x[1].seek_offset),
            ):
                assert isinstance(param, DeferredTensor)

                merge_pool = [
                    MergeTargetTensor(
                        param.materialize("cpu"), self._primary_model.merge_weight
                    )
                ]

                for model in self._secondary_models:
                    try:
                        merge_pool.append(
                            MergeTargetTensor(
                                model.parameter_index[param_name].materialize(
                                    device="cpu"
                                ),
                                model.merge_weight,
                            )
                        )
                    except KeyError:
                        print(
                            f"[!] {model.path} does not have parameter {param_name}! Skipping..."
                        )

                yield param_name, merge_tensors(merge_pool, param_name)

    def __init__(self, primary_model: Model, secondary_models: List[Model]) -> None:
        self.primary_model = primary_model
        self.secondary_models = secondary_models
        self._unpatched_load_state_dict = (
            transformers.modeling_utils._load_state_dict_into_meta_model
        )

    def __enter__(self) -> None:
        def _state_dict_override(model, state_dict, *args, **kwargs):
            return self._unpatched_load_state_dict(
                model,
                merge_with_secondary_models.PatchedStateDict(
                    state_dict, self.primary_model, self.secondary_models
                ),
                *args,
                **kwargs,
            )

        transformers.modeling_utils._load_state_dict_into_meta_model = (
            _state_dict_override
        )

    def __exit__(self, *args) -> None:
        transformers.modeling_utils._load_state_dict_into_meta_model = (
            self._unpatched_load_state_dict
        )


def get_model_keys(model_path: str) -> list:
    # Not a great solution but HF decided to no longer allow `device_map` to be
    # a string (or maybe the other way around)
    with accelerate.init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_path)
        return list(model.state_dict().keys())


class ParameterIndex:
    class index_tensors:
        def __init__(self, model: Model) -> None:
            self.model = model
            self._unpatched = (
                transformers.modeling_utils._load_state_dict_into_meta_model
            )

        def __enter__(self) -> None:
            def _cache_tensors(model, state_dict, *args, **kwargs):
                for tensor_name, deferred_tensor in state_dict.items():
                    self.model.deferred_tensors[tensor_name] = deferred_tensor
                return [], None, None

            # `_load_state_dict_into_meta_model` chosen for easy access to state_dict,
            # and I'm already familiar with it. There may be a higher-level function
            # called earlier in the callstack which could potentially reduce memory overhead.
            transformers.modeling_utils._load_state_dict_into_meta_model = (
                _cache_tensors
            )

            # Don't confuse the user with progress bars for fake models!
            transformers.utils.logging.disable_progress_bar()

        def __exit__(self, *args) -> None:
            transformers.modeling_utils._load_state_dict_into_meta_model = (
                self._unpatched
            )
            transformers.utils.logging.enable_progress_bar()

    def __init__(self, path: str):
        self.deferred_tensors = {}

        device_map = {k: "meta" for k in get_model_keys(path)}

        # This context manager writes DeferredTensors to `self.deferred_tensors`
        # instead of actually loading weights.
        with defer_tensor_load(), ParameterIndex.index_tensors(self):
            try:
                # Using the meta device saves a significant amount of memory over
                # CPU, despite no weights loading (in theory). Patching in a
                # better place level function might fix that but using the meta
                # device is an easier fix :P
                print(f"[index] Indexing '{path}'...")
                AutoModelForCausalLM.from_pretrained(
                    path, device_map=device_map, **HF_KWARGS
                )
            except ValueError as e:
                # Actual model load fails because weights aren't loaded (which is what we want)
                if "is on the meta device, we need a `value`" not in str(e):
                    raise e

    def __getitem__(self, key: str) -> DeferredTensor:
        return self.deferred_tensors[key]


class Model:
    def __init__(self, path: str, merge_weight: float) -> None:
        self.path = path
        self.merge_weight = merge_weight

        # This will only be populated in secondary models, since we use the
        # Huggingface loader completely on the primary model to make sure no
        # hooks or other stuff messes up
        self.parameter_index: Optional[ParameterIndex] = None

    def index_model_parameters(self) -> None:
        self.parameter_index = ParameterIndex(self.path)

    @staticmethod
    def from_arg(arg: str) -> Model:
        try:
            path, merge_weight = arg.split(":")
        except ValueError:
            raise BadArgException(
                "Model needs to be seperated from merge weight by colon. It should look something like this: facebook/opt-125m:0.5"
            )

        try:
            merge_weight = float(merge_weight)
        except ValueError:
            raise BadArgException(
                f"Please pass in a float as the merge weight. Got '{merge_weight}'"
            )

        return Model(path.strip(), merge_weight)


class BadArgException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__()


def main():
    global load_progress
    global DEVICE_MAP

    parser = argparse.ArgumentParser(prog="stream_merge.py")
    parser.add_argument(
        "models",
        type=str,
        nargs="+",
        help="A space-seperated list of models you want to merge, with a colon dillemeting merge ratios. Ex: 'facebook/opt-125m:0.5 notfacebook/opt-125m:0.5'",
    )
    parser.add_argument("--out", action="store", required=True)

    args = parser.parse_args()

    try:
        os.makedirs(args.out)
    except FileExistsError:
        raise BadArgException("Out dir already exists")

    if not os.path.isdir(args.out):
        raise BadArgException("Unable to find out dir")
    models = [Model.from_arg(m) for m in args.models]

    merge_sum = sum([m.merge_weight for m in models])
    if merge_sum != 1.0:
        raise BadArgException(
            f"Expected merge weight to add up to 1.0, added up to {merge_sum}"
        )

    load_progress = tqdm(desc="Read", unit="B", unit_scale=True, leave=None)

    # TODO: Check that config makes sense
    primary_model = models[0]
    secondary_models = models[1:]

    for model in secondary_models:
        model.index_model_parameters()
        # print(model.parameter_index.deferred_tensors)
    print("[index] Done indexing secondary models!")

    print(f"[merge] Merging into primary model '{primary_model.path}'...")
    if isinstance(DEVICE_MAP, str):
        DEVICE_MAP = {k: DEVICE_MAP for k in get_model_keys(primary_model.path)}

    with defer_tensor_load(), merge_with_secondary_models(
        primary_model, secondary_models
    ):
        merged_model = AutoModelForCausalLM.from_pretrained(
            primary_model.path, device_map=DEVICE_MAP, **HF_KWARGS
        )

    merged_model.save_pretrained(args.out)

    print("[save] Done! :^)")
    CheckpointChunkCache.clear(cleanup=True)

    # import psutil
    # max_mem_mib = psutil.Process().memory_info().rss / 1024**2
    # print(f"Memory usage: {round(max_mem_mib, 2)}MB")
    # input("Enter to continue...")


if __name__ == "__main__":
    # Seperate function to not wrap big amount of code in ugly indentation
    try:
        main()
    except BadArgException as e:
        print("Error:", e.message)
