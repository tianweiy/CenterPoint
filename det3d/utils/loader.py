import importlib
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("det3d.utils.loader")

CUSTOM_LOADED_MODULES = {}


def _get_possible_module_path(paths):
    ret = []
    for p in paths:
        p = Path(p)
        for path in p.glob("*"):
            if path.suffix in ["py", ".so"] or (path.is_dir()):
                if path.stem.isidentifier():
                    ret.append(path)
    return ret


def _get_regular_import_name(path, module_paths):
    path = Path(path)
    for mp in module_paths:
        mp = Path(mp)
        if mp == path:
            return path.stem
        try:
            relative_path = path.relative_to(Path(mp))
            parts = list((relative_path.parent / relative_path.stem).parts)
            module_name = ".".join([mp.stem] + parts)
            return module_name
        except Exception:
            pass
    return None


def import_file(path, name: str = None, add_to_sys=True, disable_warning=False):
    global CUSTOM_LOADED_MODULES
    path = Path(path)
    module_name = path.stem
    try:
        user_paths = os.environ["PYTHONPATH"].split(os.pathsep)
    except KeyError:
        user_paths = []
    possible_paths = _get_possible_module_path(user_paths)
    model_import_name = _get_regular_import_name(path, possible_paths)
    if model_import_name is not None:
        return import_name(model_import_name)
    if name is not None:
        module_name = name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not disable_warning:
        logger.warning(
            (
                f"Failed to perform regular import for file {path}. "
                "this means this file isn't in any folder in PYTHONPATH "
                "or don't have __init__.py in that project. "
                "directly file import may fail and some reflecting features are "
                "disabled even if import succeed. please add your project to PYTHONPATH "
                "or add __init__.py to ensure this file can be regularly imported. "
            )
        )

    if add_to_sys:  # this will enable find objects defined in a file.
        # avoid replace system modules.
        if module_name in sys.modules and module_name not in CUSTOM_LOADED_MODULES:
            raise ValueError(f"{module_name} exists in system.")
        CUSTOM_LOADED_MODULES[module_name] = module
        sys.modules[module_name] = module
    return module


def import_name(name, package=None):
    module = importlib.import_module(name, package)
    return module
