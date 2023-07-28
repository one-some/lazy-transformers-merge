import functools
import os
import pickle
import zipfile
from dataclasses import dataclass
from io import BufferedReader
from typing import Optional

import torch


class DeferredTensor:
    def __init__(
        self,
        storage_type,
        key: str,
        location: str,
    ):
        self.storage_type = storage_type
        self.key = key
        self.location = location

        # Added in `rebuild_tensor`
        self.shape: torch.Size
        self.dtype: torch.dtype
        self.stride: torch.Size
        self.seek_offset: int

        self.file_name: str

        self.size_bytes = 0

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

        self.size_bytes = nbytes

        assert isinstance(checkpoint, zipfile.ZipFile)

        active_chunk.handle.seek(self.seek_offset, os.SEEK_SET)
        storage = torch.UntypedStorage.from_buffer(
            active_chunk.handle.read(nbytes), "little", dtype=self.dtype
        )

        storage = torch.serialization._get_restore_location(device)(
            storage, self.location
        )
        tensor = torch.tensor([], dtype=self.dtype, device=storage.device)
        tensor.set_(storage, 0, self.shape, self.stride)
        tensor.requires_grad = False
        return tensor


class DeferredUnpickler(pickle.Unpickler):
    def persistent_load(self, saved_id):
        assert isinstance(saved_id, tuple)
        typename = saved_id[0]
        assert (
            typename == "storage"
        ), f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        storage_type, key, location, _ = saved_id[1:]
        return DeferredTensor(storage_type, key, location)


class defer_tensor_load:
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

    def __init__(self) -> None:
        self._unpatched_pickler = pickle.Unpickler
        self._unpatched_torch_rebuild = torch._utils._rebuild_tensor
        # HACK: Yeaaaaaaaaaah this isn't great
        defer_tensor_load._patched_load._unpatched = torch.load

    def __enter__(self) -> None:
        pickle.Unpickler = DeferredUnpickler
        torch._utils._rebuild_tensor = defer_tensor_load._patched_rebuild_tensor
        torch.load = defer_tensor_load._patched_load

    def __exit__(self, *args) -> None:
        pickle.Unpickler = self._unpatched_pickler
        torch._utils._rebuild_tensor = self._unpatched_torch_rebuild
        torch.load = defer_tensor_load._patched_load._unpatched


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
