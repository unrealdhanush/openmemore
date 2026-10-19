"""OpenMCP: one shared memory for all of a user's AI assistants (SPEC.md).

`from openmcp import Memory` gives programs the same memory, in-process (SPEC.md M7.5).
It's loaded on first use, so `openmcp serve` and the CLI don't pay for it.
"""

__version__ = "0.1.0"

__all__ = ["Hit", "Memory", "OpenMCPError", "__version__"]


def __getattr__(name: str):
    if name in ("Memory", "Hit", "OpenMCPError"):
        from openmcp import library

        return getattr(library, name)
    raise AttributeError(f"module 'openmcp' has no attribute {name!r}")
