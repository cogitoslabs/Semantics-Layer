"""
Global pytest configuration and environment patches.
"""
import importlib.metadata

_orig_meta_ver = importlib.metadata.version

def _safe_meta_version(name):
    try:
        v = _orig_meta_ver(name)
        if v is not None:
            return v
    except Exception:
        pass
    return "25.0.0"

importlib.metadata.version = _safe_meta_version
