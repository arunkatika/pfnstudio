"""Remote SSH compute adapter — BYO GPU host.

End-to-end flow when `submit()` is called:

  1. Read SSH endpoint + workspace from env (set by the API runner after
     looking up the run's compute.host_id in the ComputeHost table).
  2. Tar the project locally; scp it to <workspace>/run-<id>.tgz on the
     remote.
  3. SSH in, untar under <workspace>/run-<id>/, install priorstudio + the
     project's requirements.txt, then run the local adapter inside that
     dir with PFNSTUDIO_JSON_PROGRESS=1 so progress events flow back
     to us line-by-line.
  4. Pull results.json back; emit `remote.training_done`.
  5. Leave the host alone. We never provision, never destroy — the user
     manages the box's lifecycle (rented Vast box, personal workstation,
     corp cluster, whatever).

Why this exists alongside the vast adapter: the vast flow provisions a
fresh instance per dispatch, which costs ~2-3 minutes of wall time per
run and risks leaking instances if the API crashes mid-dispatch. For
users running many small experiments, renting a Vast box manually once
and reusing it via this adapter is operationally simpler.

Most of the SSH/scp/streaming machinery is shared with vast.py — we
import its helpers rather than duplicate them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from .base import ComputeAdapter
from .vast import (
    _emit,
    _emit_log,
    _make_project_tarball,
    _ssh_base_args,
    _wait_for_ssh_connection,
)


def _config_from_env() -> dict[str, Any]:
    return {
        "ssh_host": os.environ.get("REMOTE_SSH_HOST"),
        "ssh_user": os.environ.get("REMOTE_SSH_USER", "root"),
        "ssh_port": int(os.environ.get("REMOTE_SSH_PORT", "22")),
        "ssh_key_path": os.environ.get(
            "REMOTE_SSH_KEY_PATH",
            str(Path.home() / ".ssh" / "id_rsa"),
        ),
        "workspace": os.environ.get("REMOTE_WORKSPACE"),
        # When the host has priorstudio installed in a specific venv,
        # the user pastes that venv's python in /settings/hosts. Empty
        # means "fall back to whatever `priorstudio` is on PATH".
        "python": os.environ.get("REMOTE_PYTHON"),
        "host_name": os.environ.get("REMOTE_HOST_NAME", "remote"),
    }


# vast.py's _ssh_base_args takes a (key, host, port) triple and assumes
# user=root. The remote adapter needs to honour an arbitrary sshUser
# (ubuntu for AWS, the operator's login on personal boxes, etc), so we
# wrap that helper with a user override here.
def _remote_ssh_args(cfg: dict[str, Any]) -> list[str]:
    args = _ssh_base_args(cfg["ssh_key_path"], cfg["ssh_host"], cfg["ssh_port"])
    # The last positional in _ssh_base_args is `root@host`; replace it.
    args[-1] = f"{cfg['ssh_user']}@{cfg['ssh_host']}"
    return args


def _remote_ssh_stream(
    cfg: dict[str, Any],
    command: str,
    forward_events: bool = False,
    capture_done: list[dict[str, Any]] | None = None,
) -> int:
    """Same as vast._ssh_stream but honours sshUser. We don't reuse vast's
    helper directly because it hard-codes `root@`; copy-pasting the proc
    plumbing keeps the user-override clean.

    When `capture_done` is provided, every forwarded `done` event is
    also appended to that list — used by submit() to grab the remote
    trainer's full result dict (steps / final_loss / wall_time_s / eval
    metrics) so we can re-emit it with a local checkpoint_dir without
    dropping fields."""
    cmd = ["ssh", *_remote_ssh_args(cfg), command]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            if forward_events:
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict) and "event" in parsed:
                        sys.stdout.write(line + "\n")
                        sys.stdout.flush()
                        if capture_done is not None and parsed.get("event") == "done":
                            capture_done.append(parsed)
                        continue
                except Exception:
                    pass
                _emit("log", line=line)
            else:
                _emit("log", line=line)
    finally:
        proc.wait()
    return proc.returncode


def _remote_scp_to(cfg: dict[str, Any], local: Path, remote_path: str) -> None:
    """scp with user-override. vast._scp_to_remote assumes root@."""
    cmd = [
        "scp",
        "-i",
        cfg["ssh_key_path"],
        "-P",
        str(cfg["ssh_port"]),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        str(local),
        f"{cfg['ssh_user']}@{cfg['ssh_host']}:{remote_path}",
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"scp failed: {r.stderr.decode(errors='replace')}")


def _remote_scp_from(cfg: dict[str, Any], remote_path: str, local: Path) -> bool:
    cmd = [
        "scp",
        "-i",
        cfg["ssh_key_path"],
        "-P",
        str(cfg["ssh_port"]),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        f"{cfg['ssh_user']}@{cfg['ssh_host']}:{remote_path}",
        str(local),
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode == 0:
        return True
    err = r.stderr.decode(errors="replace")
    if "No such file" in err or "not a regular file" in err:
        return False
    raise RuntimeError(f"scp from remote failed: {err}")


def _remote_scp_dir_from(cfg: dict[str, Any], remote_dir: str, local_dir: Path) -> bool:
    """Pull an entire directory back from the remote (scp -r). Used for
    checkpoint dirs which contain model.pt + topology.json. Returns
    True on success, False if the remote dir doesn't exist (some short
    runs don't checkpoint), raises on other errors."""
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "scp",
        "-r",
        "-i",
        cfg["ssh_key_path"],
        "-P",
        str(cfg["ssh_port"]),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        f"{cfg['ssh_user']}@{cfg['ssh_host']}:{remote_dir}/.",
        str(local_dir),
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode == 0:
        return True
    err = r.stderr.decode(errors="replace")
    if "No such file" in err or "not a directory" in err:
        return False
    raise RuntimeError(f"scp -r from remote failed: {err}")


class RemoteAdapter(ComputeAdapter):
    name = "remote"

    def submit(self, run_yaml: Path, project_root: Path) -> dict[str, Any]:
        cfg = _config_from_env()
        if not cfg["ssh_host"]:
            return {
                "status": "error",
                "reason": "REMOTE_SSH_HOST not set — the API runner didn't resolve a saved host. "
                "Pick one in the run composer or add a host at /settings/hosts.",
            }
        if not cfg["workspace"]:
            return {
                "status": "error",
                "reason": "REMOTE_WORKSPACE not set — the saved host has no remote workspace path. "
                "Edit it at /settings/hosts.",
            }
        if shutil.which("ssh") is None or shutil.which("scp") is None:
            return {"status": "error", "reason": "ssh/scp not found on PATH on the API host."}
        if not os.path.exists(cfg["ssh_key_path"]):
            return {
                "status": "error",
                "reason": (
                    f"SSH key not found at {cfg['ssh_key_path']}. The runner uses the same key "
                    f"as the Vast adapter — run `./run.sh install` on the API host to bootstrap "
                    f"it, then add its .pub half to {cfg['ssh_user']}@{cfg['ssh_host']}:~/.ssh/authorized_keys."
                ),
            }

        _emit(
            "remote.starting",
            host=cfg["host_name"],
            ssh=f"{cfg['ssh_user']}@{cfg['ssh_host']}:{cfg['ssh_port']}",
            workspace=cfg["workspace"],
        )
        _emit_log(f"target host: {cfg['host_name']}")
        _emit_log(f"ssh endpoint: {cfg['ssh_user']}@{cfg['ssh_host']}:{cfg['ssh_port']}")
        _emit_log(f"ssh key:      {cfg['ssh_key_path']}")
        _emit_log(f"workspace:    {cfg['workspace']}")
        if cfg.get("python"):
            _emit_log(f"python:       {cfg['python']}")

        # Verify connectivity before bothering to tar the project. Saves
        # the user 5-30 seconds of waiting just to learn their key isn't
        # in authorized_keys.
        _emit_log("connecting via ssh…")
        try:
            _wait_for_ssh_connection(
                cfg["ssh_key_path"],
                cfg["ssh_host"],
                cfg["ssh_port"],
                timeout_s=60,
                ssh_user=cfg["ssh_user"],
            )
            _emit_log("ssh connected ✓")
        except Exception as e:
            return {
                "status": "error",
                "reason": (
                    f"Could not SSH to {cfg['ssh_user']}@{cfg['ssh_host']}:{cfg['ssh_port']}: {e}. "
                    f"If the stderr above mentions 'Permission denied (publickey)', the auto-attach "
                    f"didn't take effect — paste the API host's pub key into the instance's "
                    f"~/.ssh/authorized_keys (Composer → Show API host's SSH public key)."
                ),
            }

        # Per-dispatch subdir so concurrent runs don't trample each other.
        run_dir = f"{cfg['workspace'].rstrip('/')}/run-{uuid.uuid4().hex[:8]}"
        tarball_path = f"{run_dir}.tgz"

        _emit("remote.uploading_project", target=run_dir)
        _emit_log("packing project + priorstudio packages…")
        tar_start = time.time()
        tarball = _make_project_tarball(project_root)
        tar_size = tarball.stat().st_size
        _emit_log(
            f"tarball built: {tarball.name} ({tar_size / 1024 / 1024:.1f} MB, {time.time() - tar_start:.1f}s)"
        )
        try:
            # Ensure parent workspace exists. mkdir -p is idempotent.
            _emit_log(f"ensuring remote workspace exists: {cfg['workspace']}")
            mk_rc = _remote_ssh_stream(cfg, f"mkdir -p '{cfg['workspace']}'")
            if mk_rc != 0:
                return {
                    "status": "error",
                    "reason": f"mkdir -p on remote failed (exit {mk_rc}). Check REMOTE_WORKSPACE is writable.",
                }
            _emit_log(f"scp → {cfg['ssh_user']}@{cfg['ssh_host']}:{tarball_path}")
            scp_start = time.time()
            _remote_scp_to(cfg, tarball, tarball_path)
            _emit_log(f"upload complete ({time.time() - scp_start:.1f}s)")
        finally:
            try:
                tarball.unlink()
            except Exception:
                pass

        try:
            run_rel = run_yaml.relative_to(project_root)
            # Choose the Python executable: explicit per-host path (e.g.
            # /home/ubuntu/.venv/bin/python) or fall back to `priorstudio`
            # on PATH. The remote install step is identical either way.
            # Build the shell snippets in plain strings so we don't rely on
            # the relaxed-f-string parser (Python 3.12+); older 3.10/3.11
            # remotes would otherwise reject the install on import.
            py = cfg.get("python")
            py_cmd = f'"{py}"' if py else "python3"
            pip_cmd = f'"{py}" -m pip' if py else "pip"
            ps_cmd = f'"{py}" -m priorstudio' if py else "priorstudio"
            # Torch is required by pfnstudio-core but isn't bundled as a
            # hard dep (it's huge, GPU-specific, and the local install lets
            # the user pick CUDA vs CPU wheels). Vast templates vary —
            # most PyTorch images ship torch already, some bare CUDA
            # templates don't. Check first; install only if missing, so
            # we don't redownload 2GB every dispatch on a properly-set-up
            # instance. Default index (download.pytorch.org) auto-picks
            # the right CUDA wheel for the host.
            install_cmd = (
                f"set -euo pipefail; "
                f"mkdir -p '{run_dir}' && cd '{run_dir}' && "
                f"tar -xzf '{tarball_path}' && "
                f"if ! {py_cmd} -c 'import torch' 2>/dev/null; then "
                f"  echo '[remote] torch missing — installing CUDA wheels (one-time, ~2GB)…'; "
                f"  {pip_cmd} install --quiet torch --index-url https://download.pytorch.org/whl/cu121 "
                f"    || {pip_cmd} install --quiet torch; "
                f"fi && "
                f"{pip_cmd} install --quiet -e ./pfnstudio-core -e ./priorstudio-cli && "
                f"if [ -f ./project/requirements.txt ]; then "
                f"  {pip_cmd} install --quiet -r ./project/requirements.txt; "
                f"fi"
            )
            _emit("remote.installing")
            _emit_log(f"extracting tarball → {run_dir}")
            _emit_log("checking torch availability + installing priorstudio packages…")
            install_start = time.time()
            rc = _remote_ssh_stream(cfg, install_cmd)
            if rc != 0:
                return {
                    "status": "error",
                    "reason": f"remote install failed (exit {rc}) — see log lines above for the offending command.",
                }
            _emit_log(f"install complete ({time.time() - install_start:.1f}s)")

            run_cmd = (
                f"set -euo pipefail; cd '{run_dir}/project' && "
                f"PFNSTUDIO_JSON_PROGRESS=1 "
                f"{ps_cmd} run '{run_rel}' --target local"
            )
            _emit("remote.training_started")
            _emit_log(f"running: {ps_cmd} run {run_rel} --target local")
            _emit_log(f"  cwd: {run_dir}/project")
            start_time = time.time()
            # Capture the remote trainer's `done` event so we can re-emit
            # it with a local checkpoint_dir after the scp pull. Without
            # this, the API runner sees the remote path (which doesn't
            # exist on the API host) and skips checkpoint persistence.
            captured_done: list[dict[str, Any]] = []
            rc = _remote_ssh_stream(cfg, run_cmd, forward_events=True, capture_done=captured_done)
            if rc != 0:
                return {"status": "error", "reason": f"remote training failed (exit {rc})."}

            # Results come through the streamed `done` event — the local
            # trainer (see pfnstudio_core/training/loop.py) doesn't write
            # a separate results.json file. The streamed done was already
            # forwarded to the API and lives in run.events. Use the last
            # captured one as our results record for the return value.
            results: dict[str, Any] = (
                {k: v for k, v in captured_done[-1].items() if k not in ("event", "ts")}
                if captured_done
                else {
                    "status": "completed",
                    "reason": "remote training finished but emitted no `done` event",
                }
            )

            # Pull the checkpoint directory (model.pt + topology.json + ...).
            # Without this the run can't be used for predict / try-it after
            # the cleanup `rm -rf` further down nukes the remote dir.
            #
            # The trainer writes to <cwd>/checkpoint/ — see loop.py. Our
            # cwd on the remote is <run_dir>/project, so that resolves to
            # <run_dir>/project/checkpoint. (My earlier guess of
            # <run_dir>/project/runs/<slug>.checkpoint was wrong, which
            # is why predict/try-it stayed disabled for remote runs.)
            #
            # We re-emit the captured `done` event with the LOCAL path
            # appended so the API runner's existing checkpoint logic
            # (copy from done.checkpoint_dir → stable storage) just works.
            # The original remote-path `done` was already streamed; this
            # one overrides it because the API runner keeps the latest.
            remote_ckpt_dir = f"{run_dir}/project/checkpoint"
            local_ckpt_dir = Path(tempfile.mkdtemp(prefix="priorstudio-remote-ckpt-"))
            _emit_log(f"pulling checkpoint: {remote_ckpt_dir}")
            ckpt_start = time.time()
            try:
                got_ckpt = _remote_scp_dir_from(cfg, remote_ckpt_dir, local_ckpt_dir)
                if got_ckpt:
                    files = list(local_ckpt_dir.iterdir())
                    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
                    _emit_log(
                        f"checkpoint downloaded → {local_ckpt_dir} "
                        f"({len(files)} files, {total_bytes / 1024:.1f} KB, "
                        f"{time.time() - ckpt_start:.1f}s)"
                    )
                    if captured_done:
                        # Preserve every field from the remote trainer's
                        # done event (steps, final_loss, mean_loss_last_10pct,
                        # eval results, ...). Just swap checkpoint_dir.
                        corrected = {**captured_done[-1], "checkpoint_dir": str(local_ckpt_dir)}
                        _emit(**{k: v for k, v in corrected.items() if k != "ts"})
                    else:
                        # No streamed done to preserve — emit a minimal
                        # one so the API still picks up the checkpoint.
                        _emit("done", status="completed", checkpoint_dir=str(local_ckpt_dir))
                    results["checkpoint_dir"] = str(local_ckpt_dir)
                else:
                    _emit_log(
                        "no checkpoint dir on remote — run finished without writing one "
                        "(short runs / priors without persisted state). Predict + try-it "
                        "won't be available for this run."
                    )
            except Exception as e:
                _emit_log(f"checkpoint pull failed: {e} (predict/try-it disabled for this run)")

            elapsed = time.time() - start_time
            _emit("remote.training_done", elapsed_seconds=round(elapsed, 2))
            _emit_log(f"training elapsed: {elapsed:.1f}s")
            results.setdefault("compute", {})
            if isinstance(results.get("compute"), dict):
                results["compute"].update(
                    {
                        "provider": "remote",
                        "host_name": cfg["host_name"],
                        "elapsed_seconds": round(elapsed, 2),
                    }
                )
            return results
        finally:
            # Best-effort cleanup of the per-dispatch dir + tarball. The
            # user manages disk on their own box; leaving stale dirs on
            # crash is fine, the next run will create a new one.
            _emit_log(f"cleanup: rm -rf {run_dir} {tarball_path}")
            try:
                _remote_ssh_stream(cfg, f"rm -rf '{run_dir}' '{tarball_path}'")
            except Exception as e:
                _emit_log(f"cleanup of {run_dir} failed: {e} (non-fatal)")
