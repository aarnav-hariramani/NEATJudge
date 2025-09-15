import yaml, pickle
from pathlib import Path
from typing import Any, Dict

def read_yaml(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())

def ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def save_pickle(obj: Any, path: str):
    ensure_dir(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
