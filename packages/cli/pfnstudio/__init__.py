# Single source of truth for the version is packages/cli/pyproject.toml.
# Read it back from the installed distribution metadata so __version__
# can never drift from what pip actually installed (a hardcoded literal
# here silently reported a stale version for several releases).
try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("pfnstudio")
    except PackageNotFoundError:
        # Running from a source tree with no installed dist-info.
        __version__ = "0.0.0+dev"
except Exception:  # pragma: no cover — importlib.metadata always present on 3.8+
    __version__ = "0.0.0+unknown"
