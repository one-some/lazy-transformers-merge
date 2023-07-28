from __future__ import annotations

from typing import Optional

from modeling.parameter_index import ParameterIndex

loaded_models = []


def load_model(path: str) -> Model:
    model = Model(path)
    loaded_models.append(model)
    return model


class Model:
    def __init__(self, path: str) -> None:
        self.path = path
        self.merge_weight = 0.0

        self.parameter_index = ParameterIndex(path)

    def index_model_parameters(self) -> None:
        self.parameter_index = ParameterIndex(self.path)

    def structure_json(self):
        layers_key = "model.decoder.layers"
        ret = {
            "path": self.path,
            "parameters": [],
        }
        for param_name, param in self.parameter_index.deferred_tensors.items():
            layer_index = None
            if layers_key in param_name:
                layer_index = int(param_name.replace(layers_key, "").strip(".").split(".", 1)[0])

            ret["parameters"].append(
                {
                    "name": param_name,
                    "shape": param.shape,
                    "dtype": str(param.dtype),
                    "sizeBytes": param.size_bytes,
                    "layerIndex": layer_index
                }
            )

        return ret
