"""Self-hosted runner — pull jobs from PFN Studio cloud and run them locally.

The runner pattern mirrors GitHub Actions' self-hosted runner: outbound-
only long-poll, no inbound port opens on the user's machine. Works
behind NAT/firewall, no SSH key dance, registers once and stays
addressable across reboots.

Wire protocol:

  POST /runner/capabilities                  on startup: report torch/CUDA/disk
  GET  /runner/jobs/next                     long-poll for next claimable job
  POST /runner/jobs/:runId/heartbeat         every HEARTBEAT_S seconds
  POST /runner/jobs/:runId/result            final upload (status + events)

Auth header: ``Authorization: Token psr_<48-hex>``. Token issued exactly
once when a user clicks "Add a runner" in /settings/runners; stored at
``~/.pfnstudio/runner.json`` with chmod 0600.

Phase 2 scope (this file):
  * register / doctor / start subcommands
  * Long-poll loop with SIGINT / SIGTERM graceful shutdown
  * Per-job heartbeat thread
  * Subprocess execution via ``pfnstudio run``, with JSON-event piping
    back to the cloud as Run.events

Out of scope for Phase 2 (will land in 2.5):
  * Bundle download (``GET /runner/jobs/:id/bundle``) for jobs whose
    project isn't already on this runner's disk. Today the runner
    assumes ``--project-root`` (default: CWD) contains the project the
    job references. Works for the developer-on-own-machine flow; not
    yet for "studio dispatches a job to my laptop blindly".
  * Per-job venv isolation (we run in the runner's own Python env).
  * Docker executor mode.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import typer
import yaml
from rich.console import Console
from rich.table import Table

from . import __version__

console = Console()

# Where the bearer token + cloud URL live. Permissions are 0600 so other
# users on the same machine can't steal the token.
CONFIG_PATH = Path.home() / ".pfnstudio" / "runner.json"

# How often we tell the cloud "yes I'm still running this job". Has to
# be shorter than the cloud's HEARTBEAT_STALE_S threshold (90s) so the
# job's claim doesn't expire mid-training.
HEARTBEAT_INTERVAL_S = 30

# Long-poll request timeout. The cloud responds within ~25s; we give it
# 35s of slack to cover network latency + reverse-proxy buffering.
POLL_TIMEOUT_S = 35

# Backoff schedule when the cloud is unreachable / 5xx. Capped so a
# transient blip doesn't push the next attempt arbitrarily far.
BACKOFF_LADDER_S = (2, 5, 15, 30, 60)

runner_app = typer.Typer(
    name="runner",
    help="Self-hosted runner — register this machine and pull jobs from PFN Studio.",
    no_args_is_help=True,
)


# ── Config helpers ────────────────────────────────────────────────────


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        console.print(
            "[red]Not registered.[/red] Run [bold]pfnstudio runner register --token psr_...[/bold]"
        )
        raise typer.Exit(2)
    try:
        return json.loads(CONFIG_PATH.read_text())
    except json.JSONDecodeError as e:
        console.print(f"[red]Corrupt config at {CONFIG_PATH}:[/red] {e}")
        raise typer.Exit(2) from e


def _save_config(cfg: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, sort_keys=True))
    # Token must NOT be world-readable — chmod after write so the file
    # never briefly has loose perms.
    try:
        CONFIG_PATH.chmod(0o600)
    except OSError:
        # On Windows the chmod is best-effort; the token is still in the
        # user's profile directory which has reasonable defaults.
        pass


# ── Capability reporting ──────────────────────────────────────────────


def _capabilities() -> dict[str, Any]:
    """Self-report what this machine can run. Posted on register + start
    so the cloud's job dispatcher can gate incompatible jobs."""
    caps: dict[str, Any] = {
        "osArch": f"{platform.system().lower()}/{platform.machine()}",
        "pythonVersion": platform.python_version(),
        "pfnstudioVersion": __version__,
        "hostname": socket.gethostname(),
    }
    # torch is optional at install time (`pip install pfnstudio[torch]`).
    # Missing torch means training will fail; we still register so the
    # operator can see "this runner reported no torch" in /settings.
    try:
        import torch  # type: ignore

        caps["torchVersion"] = torch.__version__
        caps["cudaAvailable"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                caps["cudaVersion"] = torch.version.cuda
                caps["gpu"] = torch.cuda.get_device_name(0)
                caps["gpuMemoryGb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    1,
                )
            except Exception:
                # Any of these can raise on weird CUDA setups; capability
                # reporting must never crash the runner.
                pass
    except ImportError:
        caps["torchVersion"] = None
        caps["cudaAvailable"] = False
    # Free disk on the workspace partition (assume $HOME for now).
    try:
        free = shutil.disk_usage(Path.home()).free
        caps["diskFreeGb"] = round(free / (1024**3), 1)
    except OSError:
        pass
    return caps


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ── register ─────────────────────────────────────────────────────────


@runner_app.command()
def register(
    token: str = typer.Option(..., "--token", "-t", help="Runner token (starts with psr_)."),
    cloud_url: str = typer.Option(
        "https://api.pfnstudio.com",
        "--cloud-url",
        help="Base URL of the PFN Studio API (override for staging or self-hosted).",
    ),
) -> None:
    """Save the runner token + cloud URL and verify connectivity.

    The token is issued in /settings/runners → Add a runner → copy the
    token from the modal. Stored at ~/.pfnstudio/runner.json with 0600.
    Verifies the token by posting current capabilities to the cloud.
    """
    if not token.startswith("psr_"):
        console.print(
            "[red]Bad token format.[/red] Runner tokens start with 'psr_'. "
            "Did you copy a user API token (ps_…) by mistake?"
        )
        raise typer.Exit(2)

    cloud_url = cloud_url.rstrip("/")
    caps = _capabilities()
    try:
        r = requests.post(
            f"{cloud_url}/runner/capabilities",
            json={"capabilities": caps},
            headers={"Authorization": f"Token {token}"},
            timeout=15,
        )
    except requests.RequestException as e:
        console.print(f"[red]Could not reach {cloud_url}:[/red] {e}")
        raise typer.Exit(2) from e

    if r.status_code == 401:
        console.print(
            "[red]Token rejected.[/red] Either the runner was revoked, or this "
            "token is a user API token (ps_…) not a runner token (psr_…)."
        )
        raise typer.Exit(2)
    if not r.ok:
        console.print(f"[red]Cloud returned {r.status_code}:[/red] {r.text[:200]}")
        raise typer.Exit(2)

    _save_config({"token": token, "cloud_url": cloud_url})
    console.print(f"[green]✓ Registered.[/green] Token saved to [dim]{CONFIG_PATH}[/dim]")
    console.print("[dim]Reported capabilities:[/dim]")
    for k, v in caps.items():
        console.print(f"  [dim]{k}[/dim] = {v}")
    console.print()
    console.print("Start the long-poll loop with: [bold]pfnstudio runner start[/bold]")


# ── doctor ───────────────────────────────────────────────────────────


@runner_app.command()
def doctor() -> None:
    """Preflight checks — torch, CUDA, disk, network reachability."""
    caps = _capabilities()
    table = Table(title="Runner environment", show_header=False, box=None)
    table.add_column("key", style="dim")
    table.add_column("value")
    for k, v in caps.items():
        table.add_row(k, str(v))
    console.print(table)

    problems: list[str] = []
    if not caps.get("torchVersion"):
        problems.append(
            "torch not installed. Install with: [bold]pip install 'pfnstudio[torch]'[/bold]"
        )
    if not caps.get("cudaAvailable"):
        problems.append(
            "[yellow]No CUDA[/yellow] — runner can only execute CPU-feasible runs. "
            "GPU jobs dispatched to this runner will fail at training start."
        )
    disk_gb = caps.get("diskFreeGb")
    if isinstance(disk_gb, (int, float)) and disk_gb < 10:
        problems.append(
            f"Low disk: {disk_gb} GB free. Training writes checkpoints (often "
            "hundreds of MB); 10+ GB recommended."
        )

    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
        except Exception:
            cfg = None
        if cfg and cfg.get("cloud_url"):
            console.print(f"\n[dim]Pinging cloud at[/dim] {cfg['cloud_url']}…")
            try:
                # /health is unauth so we don't even need the token here.
                r = requests.get(f"{cfg['cloud_url']}/health", timeout=10)
                if r.ok:
                    console.print(f"[green]✓ Cloud reachable[/green] ({r.status_code})")
                else:
                    problems.append(f"Cloud responded {r.status_code} on /health")
            except requests.RequestException as e:
                problems.append(f"Cloud unreachable: {e}")
    else:
        console.print(
            f"\n[dim]No config at {CONFIG_PATH}.[/dim] "
            "Run [bold]pfnstudio runner register --token ...[/bold] first."
        )

    if problems:
        console.print()
        for p in problems:
            console.print(f"  [yellow]⚠[/yellow] {p}")
        raise typer.Exit(1 if any("cuda" not in p.lower() for p in problems) else 0)
    console.print("\n[green]All checks passed.[/green]")


# ── start ────────────────────────────────────────────────────────────


@runner_app.command()
def start(
    project_root: Path = typer.Option(
        Path.cwd(),
        "--project-root",
        help=(
            "Directory containing the project the runner trains. Currently "
            "the runner doesn't fetch projects from the cloud (Phase 2.5); "
            "it runs against whatever's at this path. Default: current dir."
        ),
    ),
    poll_idle_log_every_s: int = typer.Option(
        300,
        "--idle-log-every",
        help=(
            "Print 'still polling, nothing to do' every N seconds when the "
            "queue is empty. Set to 0 to suppress idle logs entirely."
        ),
    ),
) -> None:
    """Long-poll the cloud for jobs and execute them locally.

    Press Ctrl-C to stop gracefully (current job finishes; next poll
    skipped). SIGTERM (sent by systemd / docker stop) behaves the same.
    """
    cfg = _load_config()
    base = cfg["cloud_url"]
    headers = {"Authorization": f"Token {cfg['token']}"}

    # Bump capabilities on start so a runner that was offline for a
    # week and just got an OS update reports the new state.
    try:
        requests.post(
            f"{base}/runner/capabilities",
            json={"capabilities": _capabilities()},
            headers=headers,
            timeout=15,
        )
    except requests.RequestException as e:
        console.print(f"[yellow]Could not post capabilities at startup:[/yellow] {e}")

    stop_flag = threading.Event()

    def _on_signal(signum: int, _frame: Any) -> None:
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        if not stop_flag.is_set():
            console.print(
                f"\n[yellow]Received {sig_name} — finishing current job, then stopping.[/yellow]"
            )
            stop_flag.set()
        else:
            # Second signal — exit hard.
            console.print(f"\n[red]Second {sig_name} — exiting immediately.[/red]")
            sys.exit(1)

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)

    console.print(
        f"[green]✓ Runner online.[/green] Polling [bold]{base}[/bold] for jobs.\n"
        f"[dim]Project root: {project_root}  ·  Ctrl-C to stop.[/dim]"
    )

    backoff_idx = 0
    last_idle_log = time.time()

    while not stop_flag.is_set():
        try:
            r = requests.get(
                f"{base}/runner/jobs/next",
                headers=headers,
                timeout=POLL_TIMEOUT_S,
            )
        except requests.RequestException as e:
            wait = BACKOFF_LADDER_S[min(backoff_idx, len(BACKOFF_LADDER_S) - 1)]
            console.print(f"[yellow]Poll failed:[/yellow] {e} — retrying in {wait}s")
            backoff_idx += 1
            if stop_flag.wait(wait):
                break
            continue
        backoff_idx = 0

        if r.status_code == 401:
            console.print(
                "[red]Auth failed (401).[/red] This runner was revoked — "
                "re-register from /settings/runners."
            )
            break
        if r.status_code >= 500:
            wait = BACKOFF_LADDER_S[min(backoff_idx, len(BACKOFF_LADDER_S) - 1)]
            console.print(f"[yellow]Cloud returned {r.status_code}; backing off {wait}s[/yellow]")
            backoff_idx += 1
            if stop_flag.wait(wait):
                break
            continue
        if not r.ok:
            console.print(f"[red]Cloud returned {r.status_code}:[/red] {r.text[:200]}")
            break

        data = r.json()
        if data.get("status") == "idle":
            if poll_idle_log_every_s > 0 and (time.time() - last_idle_log) >= poll_idle_log_every_s:
                console.print("[dim]· still polling, no jobs queued[/dim]")
                last_idle_log = time.time()
            continue

        job = data.get("job") or {}
        if not job.get("runId"):
            console.print(f"[red]Malformed job payload:[/red] {data}")
            continue

        _run_job(base, headers, job, project_root, stop_flag)
        last_idle_log = time.time()

    console.print("[green]Runner stopped.[/green]")


# ── Job execution ────────────────────────────────────────────────────


def _run_job(
    base: str,
    headers: dict[str, str],
    job: dict[str, Any],
    project_root: Path,
    stop_flag: threading.Event,
) -> None:
    """Execute one claimed job — subprocess the local trainer, stream
    its JSON events back via /events, post final result."""
    run_id = str(job["runId"])
    run_slug = job.get("runSlug", run_id)
    console.print(f"\n[bold blue]→ claimed[/bold blue] {run_slug} [dim](run_id={run_id})[/dim]")

    # Heartbeat thread keeps the cloud's claim fresh while training runs.
    alive = threading.Event()
    alive.set()

    def _heartbeat() -> None:
        while alive.is_set():
            try:
                requests.post(
                    f"{base}/runner/jobs/{run_id}/heartbeat",
                    headers=headers,
                    timeout=10,
                )
            except requests.RequestException:
                # Transient — keep ticking; the cloud will reclaim the job
                # if heartbeats lapse beyond HEARTBEAT_STALE_S.
                pass
            for _ in range(HEARTBEAT_INTERVAL_S):
                if not alive.is_set():
                    return
                time.sleep(1)

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    events_acc: list[dict[str, Any]] = []
    final_status = "failed"
    final_error: dict[str, Any] | None = None
    proc: subprocess.Popen[str] | None = None

    try:
        # Materialise the job's spec as a temp run.yaml. The local
        # adapter (run via the same CLI subprocess) reads that file.
        spec = {
            "id": run_slug,
            "prior": job.get("priorRef") or {},
            "model": job.get("modelRef") or {},
            "evals": job.get("evalRefs") or [],
            "hyperparams": job.get("hyperparams") or {},
            # Force local target so the dispatcher doesn't bounce back
            # to cloud — we're already on the runner's machine.
            "compute": {"target": "local"},
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=f"-{run_slug}.yaml", delete=False, dir=str(project_root)
        ) as f:
            yaml.safe_dump(spec, f)
            tmp_yaml = Path(f.name)

        # PFNSTUDIO_JSON_PROGRESS=1 tells the local adapter to emit
        # JSON-line events on stdout, which we pipe up to /events.
        env = os.environ.copy()
        env["PFNSTUDIO_JSON_PROGRESS"] = "1"

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pfnstudio",
                "run",
                str(tmp_yaml),
                "--target",
                "local",
            ],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            # Try to parse a JSON event; otherwise treat as a plain log line.
            try:
                ev = json.loads(line)
                if not isinstance(ev, dict) or "event" not in ev:
                    raise ValueError("not an event")
            except (json.JSONDecodeError, ValueError):
                ev = {"event": "log", "line": line, "ts": _ts()}
            events_acc.append(ev)

            # Best-effort stream — don't block the trainer if the cloud
            # is briefly unreachable.
            try:
                requests.post(
                    f"{base}/runner/jobs/{run_id}/events",
                    json={"events": [ev]},
                    headers=headers,
                    timeout=5,
                )
            except requests.RequestException:
                pass

            if ev.get("event") == "finished":
                final_status = ev.get("status", "completed")

            if stop_flag.is_set():
                # User asked the runner to stop; kill this job and
                # report cancelled.
                console.print("[yellow]Stop requested — terminating in-flight training.[/yellow]")
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                final_status = "cancelled"
                break

        proc.wait()
        # If the trainer exited cleanly but didn't emit a 'finished'
        # event, infer success from the return code.
        if proc.returncode == 0 and final_status == "failed":
            final_status = "completed"
        elif proc.returncode != 0 and final_status not in ("cancelled", "failed"):
            final_status = "failed"
            final_error = {"message": f"trainer exited with code {proc.returncode}"}

    except Exception as e:
        final_error = {"message": f"runner-side error: {type(e).__name__}: {e}"}
        if proc is not None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    finally:
        alive.clear()
        # Best-effort cleanup of the temp run.yaml.
        try:
            tmp_yaml.unlink(missing_ok=True)  # type: ignore[possibly-undefined]
        except Exception:
            pass

        # Post the final state. Retry once on transient errors — losing
        # this report would leave the cloud with the run stuck in
        # 'running' until the next requeue sweep.
        for attempt in range(2):
            try:
                requests.post(
                    f"{base}/runner/jobs/{run_id}/result",
                    json={
                        "status": final_status,
                        "events": events_acc,
                        "error": final_error,
                    },
                    headers=headers,
                    timeout=30,
                )
                break
            except requests.RequestException as e:
                if attempt == 0:
                    console.print(f"[yellow]Failed to post result, retrying:[/yellow] {e}")
                    time.sleep(2)
                else:
                    console.print(f"[red]Could not post result for {run_id}:[/red] {e}")

        marker = (
            "[green]✓[/green]"
            if final_status == "completed"
            else "[yellow]⚠[/yellow]"
            if final_status == "cancelled"
            else "[red]✗[/red]"
        )
        console.print(f"{marker} run {run_slug} → [bold]{final_status}[/bold]")
