import importlib
from pathlib import Path

__all__ = []

current_dir = Path(__file__).parent

for py_file in current_dir.glob("*.py"):
    if py_file.stem == "__init__":
        continue

    module = importlib.import_module(f".{py_file.stem}", package=__package__)

    globals()[py_file.stem] = module

    __all__.append(py_file.stem)
