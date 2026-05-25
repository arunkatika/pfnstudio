"""Project scaffolding — copy the FM project template and substitute placeholders."""

from __future__ import annotations

import shutil
from pathlib import Path

from ._paths import fm_project_template


def scaffold_project(target: Path, project_name: str, description: str, org: str) -> None:
    src = fm_project_template()
    shutil.copytree(src, target)

    substitutions = {
        "{{project_name}}": project_name,
        "{{one_line_description}}": description,
        "{{org}}": org,
    }

    for path in target.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".bin", ".pt", ".ckpt"}:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for token, value in substitutions.items():
            text = text.replace(token, value)
        path.write_text(text)
