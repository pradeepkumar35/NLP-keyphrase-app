"""Loads all project source files into one shared namespace.

The original code was written for a single Kaggle notebook where every class
and constant lived in the same global namespace, so the modules reference each
other through bare names and globals() lookups. Executing them into a single
shared namespace reproduces that environment and lets them resolve correctly.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

NAMESPACE = {"__file__": __file__, "__name__": "pipeline"}

_FILES = [
    "domain_keywords_expansion.py",
    "domain_fallback_classifier.py",
    "Abstractive_code.py",
    "Extractive_code.py",
    "Fusion.py",
    "contextual_keyphrase_expansion.py",
]

for _fname in _FILES:
    _path = os.path.join(_HERE, _fname)
    with open(_path, encoding="utf-8") as _f:
        _code = _f.read()
    exec(compile(_code, _path, "exec"), NAMESPACE)

globals().update(NAMESPACE)
