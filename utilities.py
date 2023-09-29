import os
import importlib
from typing import List


def open_txt_file(file_path: str) -> List[str]:
    ret = []
    with open(file_path, "r") as f:
        ret = f.readlines()
    ret = [s.strip("\n") for s in ret]
    return ret


def list_subdir(path: str):
    return sorted([folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))])


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config.target)(**config.get("params", dict()))


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)