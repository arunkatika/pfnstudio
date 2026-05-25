"""Built-in architecture blocks. Imported eagerly to register them on package load."""

from . import heads, tabular, transformer  # noqa: F401  (registers on import)
