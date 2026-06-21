"""HuggingFace push / pull helpers for the Deployments flow.

Two operations:

  - push_folder(local_dir, repo_id, token, revision='main', private=True)
      Mirror the contents of `local_dir` into `repo_id` on HF Hub. Used
      by NestJS at promote time to publish the trained run's project
      tree + checkpoint to the user's HF account.

  - pull_snapshot(repo_id, dest_dir, token, revision='main')
      Materialise a copy of `repo_id` at `revision` into `dest_dir`. Used
      by the worker at boot time when artifactRef.kind == 'hf'.

Both go through huggingface_hub so authentication, chunking, LFS
handling, and resume-on-error all work without us reimplementing them.

The CLI wraps each in a subcommand. NestJS invokes the CLI rather than
embedding Python — same pattern as `pfnstudio serve`.
"""

from __future__ import annotations

from pathlib import Path


def push_folder(
    *,
    local_dir: Path,
    repo_id: str,
    token: str,
    revision: str = "main",
    private: bool = True,
    commit_message: str | None = None,
) -> dict[str, str]:
    """Push every file under ``local_dir`` to ``repo_id``.

    Returns ``{"commit_sha": "...", "url": "..."}``.

    The repo must already exist; the NestJS side ensures that via the
    Hub's create-repo endpoint before calling here so we keep this helper
    focused. We do still pass ``create_pr=False`` so a freshly-created
    empty repo accepts the upload as a normal commit on the branch.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    # exist_ok=True so re-running the push (e.g. user clicks Promote
    # twice without deleting the deployment) doesn't blow up.
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    result = api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        commit_message=commit_message or "pfnstudio: publish trained run",
    )
    # CommitInfo has .commit_url and .oid (the commit hash). Fall back
    # to string-only if a future hub release changes the shape.
    commit_sha = getattr(result, "oid", "") or ""
    url = getattr(result, "commit_url", "") or f"https://huggingface.co/{repo_id}"
    return {"commit_sha": str(commit_sha), "url": str(url)}


def pull_snapshot(
    *,
    repo_id: str,
    dest_dir: Path,
    token: str,
    revision: str = "main",
) -> Path:
    """Download ``repo_id`` at ``revision`` into ``dest_dir``.

    Uses ``snapshot_download`` with ``local_dir=dest_dir`` so files
    end up as real files (not symlinks into the global hub cache).
    Returns ``dest_dir`` so callers can pipe straight into the worker.
    """
    from huggingface_hub import snapshot_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        token=token,
        local_dir=str(dest_dir),
    )
    return dest_dir
