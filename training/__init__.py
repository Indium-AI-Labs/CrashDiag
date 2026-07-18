"""Dataset generation and training entry points for CrashDiag.

The package deliberately has no import-time dependency on PyTorch, TRL,
Transformers, or datasets.  Those libraries are imported only by the training
commands after their arguments have been parsed.
"""

__all__: list[str] = []
