## streaming / lazy transformers merge
A script that merges transformers models without using a ton of memory.

Rather than memory usage scaling `M*n` with the amount of models merged, it's held at a constant `M`, other than a few MB of overhead for indexing parameters and caching. This is done by patching tensors to be lazy representations of their location on disk rather than their contents. These deferred/lazy tensors remain as indexes until weights are actually moved from the state dict into the model, upon which we look up all the parameters with matching keys, and merge them.

Loading the final merge into RAM could probably be avoided if I knew a bit more about torch serialization. That would result in RAM usage being a constant few MB for any model size, but as of now it's not really a huge deal since you can split the final merge across many devices (see `DEVICE_MAP` in `stream_merge.py`)

The merge algorithm is a simple weighted average, but can definitly be expanded in the future. I've put the `merge_tensors()` function in the top of the script for anyone who wants to play around :^)

### Usage
```sh
python stream_merge.py facebook/opt-125m:0.5 facebook/opt-125m:0.5 --out super_exciting_innovative_merge
```
Notes:
* Infinite(?) models can be added. They must be formatted like `[huggingface repo/local folder]:[merge weight]`, as shown above.
* Merge weights must sum to one.
* No tokenizer is saved with the model, so you'll have to load your own for inference

Tested with (packages are required but versions might be flexible):
- `transformers==4.31.0`
- `acclerate==0.21.0`
- `torch==2.0.1`
- `tqdm`

Attribution:
    - Tensor deferring trick and some materialization code from [KoboldAI](https://github.com/KoboldAI/KoboldAI-Client)'s lazyloader (AGPL-3.0)
