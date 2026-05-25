"""Registry-dataset loader.

The PFN Studio API resolves each `registry:<id>@<version>` reference an
eval names into an absolute on-disk path and writes a `.datasets/index.yaml`
into the exported project directory before spawning the CLI. This module
reads that index and lazy-loads the referenced files on demand, so eval
scorers can ask for "m4-monthly@1.0.0/train" and get a pandas DataFrame
(or raw bytes, depending on file extension) without caring where the data
physically lives.

Index file layout:

    # .datasets/index.yaml
    version: 1
    datasets:
      m4-monthly@1.0.0:
        dir: /abs/path/uploads/datasets/<orgId>/m4-monthly/1.0.0
        filename: Monthly-train.csv
        sha256: <hex>
        splits:
          - name: train
            path: Monthly-train.csv
            numRows: 48000

Resolution is best-effort — if a dataset is referenced but not in the
index (e.g. dependency was optional and not downloaded), `load()` raises
``DatasetUnavailable`` with a clear message rather than crashing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── Exceptions ───────────────────────────────────────────────────────────


class DatasetUnavailable(Exception):  # noqa: N818  # public API; renaming would break callers
    """Raised when a registry reference can't be resolved against the index."""


# ── Index parsing ────────────────────────────────────────────────────────


@dataclass
class IndexedSplit:
    name: str
    path: str
    num_rows: int | None = None


@dataclass
class IndexedDataset:
    key: str  # e.g. "m4-monthly@1.0.0"
    dir: Path  # absolute directory containing the files
    filename: str  # main downloaded artifact filename
    sha256: str | None
    splits: list[IndexedSplit]

    def split_path(self, split_name: str) -> Path:
        for s in self.splits:
            if s.name == split_name:
                return self.dir / s.path
        # Fall back to the main filename if no split matches.
        return self.dir / self.filename


def load_index(project_root: Path | str) -> dict[str, IndexedDataset]:
    """Read `.datasets/index.{json,yaml}` if present; return key→IndexedDataset.

    Returns an empty dict if the index doesn't exist — that's the common
    case for runs that don't reference any registry datasets. The API
    writes JSON for portability (no PyYAML dep); we accept either.
    """
    root = Path(project_root)
    json_path = root / ".datasets" / "index.json"
    yaml_path = root / ".datasets" / "index.yaml"

    if json_path.exists():
        raw = json_path.read_text(encoding="utf-8")
        try:
            data: Any = json.loads(raw)
        except Exception as e:
            raise DatasetUnavailable(f"Could not parse {json_path}: {e}") from e
    elif yaml_path.exists():
        raw = yaml_path.read_text(encoding="utf-8")
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(raw)
        except ImportError as e:
            raise DatasetUnavailable(
                f"{yaml_path} requires PyYAML. Install pyyaml or have the API "
                "emit .datasets/index.json instead."
            ) from e
        except Exception as e:
            raise DatasetUnavailable(f"Could not parse {yaml_path}: {e}") from e
    else:
        return {}

    out: dict[str, IndexedDataset] = {}
    for key, val in (data.get("datasets") or {}).items():
        splits = [
            IndexedSplit(name=s["name"], path=s["path"], num_rows=s.get("numRows"))
            for s in (val.get("splits") or [])
        ]
        out[key] = IndexedDataset(
            key=key,
            dir=Path(val["dir"]),
            filename=val.get("filename", ""),
            sha256=val.get("sha256"),
            splits=splits,
        )
    return out


# ── Public loader ────────────────────────────────────────────────────────


_REGISTRY_RE = re.compile(r"^registry:([^@]+)@(.+)$")


def parse_source(source: str) -> tuple[str, str] | None:
    """Parse 'registry:<id>@<version>' → (id, version). Returns None on no match."""
    m = _REGISTRY_RE.match(source.strip())
    if not m:
        return None
    return m.group(1), m.group(2)


class RegistryDatasetLoader:
    """Lazy loader over the resolved registry index.

    Usage::

        loader = RegistryDatasetLoader.from_project(project_root)
        df = loader.load_table("registry:m4-monthly@1.0.0", split="train")
    """

    def __init__(self, index: dict[str, IndexedDataset]):
        self._index = index

    @classmethod
    def from_project(cls, project_root: Path | str) -> RegistryDatasetLoader:
        return cls(load_index(project_root))

    @property
    def available(self) -> list[str]:
        """Keys (id@version) the loader can resolve."""
        return sorted(self._index.keys())

    def resolve(self, source: str) -> IndexedDataset:
        """Return the IndexedDataset for a registry source, or raise."""
        parsed = parse_source(source)
        if parsed is None:
            raise DatasetUnavailable(
                f"Not a registry source: '{source}'. Expected 'registry:<id>@<version>'."
            )
        key = f"{parsed[0]}@{parsed[1]}"
        if key not in self._index:
            avail = ", ".join(self.available) or "<none downloaded>"
            raise DatasetUnavailable(
                f"Dataset '{key}' not in run index. Available: {avail}. "
                f"Download it via the studio's /datasets page or include it in the run's "
                f"datasetDeps."
            )
        return self._index[key]

    def path(self, source: str, split: str | None = None) -> Path:
        """Resolve a source (+ optional split) to an absolute file path."""
        ds = self.resolve(source)
        if split is None:
            # Default to the main filename.
            return ds.dir / ds.filename if ds.filename else ds.split_path("")
        return ds.split_path(split)

    def load_bytes(self, source: str, split: str | None = None) -> bytes:
        """Raw bytes — escape hatch for non-tabular data."""
        return self.path(source, split).read_bytes()

    def load_text(self, source: str, split: str | None = None, encoding: str = "utf-8") -> str:
        return self.path(source, split).read_text(encoding=encoding)

    def load_table(self, source: str, split: str | None = None) -> Any:
        """Load the file as a pandas DataFrame, picking the reader by extension.

        Supports .csv, .tsv, .parquet, .json, .jsonl. Raises if pandas isn't
        installed (which is unusual for ML projects but possible in slim CI).
        """
        path = self.path(source, split)
        try:
            import pandas as pd  # type: ignore
        except ImportError as e:
            raise DatasetUnavailable(
                "load_table requires pandas. Install pfnstudio-core[data] or pip install pandas."
            ) from e

        ext = path.suffix.lower()
        if ext in (".csv",):
            return pd.read_csv(path)
        if ext in (".tsv",):
            return pd.read_csv(path, sep="\t")
        if ext == ".parquet":
            return pd.read_parquet(path)
        if ext == ".jsonl":
            return pd.read_json(path, lines=True)
        if ext == ".json":
            return pd.read_json(path)
        raise DatasetUnavailable(
            f"Don't know how to read {path.name} as a table. Supported extensions: "
            "csv, tsv, parquet, json, jsonl. Use load_bytes/load_text for other formats."
        )


__all__ = [
    "DatasetUnavailable",
    "IndexedDataset",
    "IndexedSplit",
    "RegistryDatasetLoader",
    "load_index",
    "parse_source",
]
