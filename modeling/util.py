import accelerate
from transformers import AutoModelForCausalLM


def get_model_keys(model_path: str) -> list:
    # Not a great solution but HF decided to no longer allow `device_map` to be
    # a string (or maybe the other way around)
    with accelerate.init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_path)
        return list(model.state_dict().keys())
