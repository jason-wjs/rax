"""Runs a registered bundle in a bubblewrap sandbox: no network, no host sockets, credentials,
simulator or GPU; only the system, the agent's Python environment and the bundle's workspace."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import resource
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time

from .storage import RUNTIME_SOURCE, snapshot, project_directory
from .gemini import MAX_REQUEST_BYTES
from .errors import public_failure
from harness.agent_host import thread_variables


class ExecutionDeadline(BaseException):
    """Cannot be swallowed by a tool's ordinary exception-to-MCP-error handler."""


@contextmanager
def deadline(seconds):
    """Bound even a blocked native MCP/provider call, not just the sandbox's CPU time. None sets no
    limit of its own, but a timer set inside the block (a finalization limit) still raises."""
    def timeout(*_):
        raise ExecutionDeadline("Execution wall-time budget exhausted")
    old = signal.signal(signal.SIGALRM, timeout)
    previous = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    if seconds is not None:
        signal.setitimer(signal.ITIMER_REAL, min(seconds, previous[0]) if previous[0] else seconds)
    try:
        yield
    finally:
        remaining = max(0.000001, previous[0] - (time.monotonic() - started)) if previous[0] else 0
        signal.setitimer(signal.ITIMER_REAL, remaining, previous[1])
        signal.signal(signal.SIGALRM, old)


def wall_end(limit, seconds=None):
    """When a run must end: the tighter of its limit and the time left in its episode, as a
    monotonic time; None when neither is set (the episode ends at its step limit)."""
    bounds = [value for value in (limit, seconds) if value is not None]
    return time.monotonic() + min(bounds) if bounds else None


def time_left(end_at):
    return None if end_at is None else end_at - time.monotonic()


def finalize_within(seconds):
    """After the episode ends, give the run at most `seconds` more, within any limit left."""
    remaining, _ = signal.getitimer(signal.ITIMER_REAL)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, remaining) if remaining else seconds)


# The host system the bundle's Python needs, read-only (merged-/usr links are recreated).
SYSTEM = ('/usr', '/etc', '/opt')
LINKS = ('/bin', '/sbin', '/lib', '/lib32', '/lib64', '/libx32')
OPEN_FILES = 128


def python_environment():
    """The agent's Python environment (the interpreter's prefix), by its real path."""
    return Path(sys.prefix).resolve()


def output_bytes(directory, limit):
    """Bytes under `directory`, counting no further than just past `limit`."""
    total, pending = 0, [str(directory)]
    while pending and total <= limit:
        try:
            entries = list(os.scandir(pending.pop()))
        except OSError:
            continue
        for entry in entries:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISDIR(info.st_mode):
                pending.append(entry.path)
            total += info.st_blocks*512
    return total


class BundleSandbox:
    # Seconds allowed for the process to exit after the episode ends.
    finalization_seconds = 10

    def __init__(self, limits):
        self.limits = limits
        self.execution_seconds = 0.0

    def command(self, bundle, mode, entrypoint, *, workspace, layout, writable, sdk, frames=None,
                workspace_path='code/controller'):
        """The bundle's Python at the agent's own workspace path: a read-only scaffold with
        the bundle at its project path, the runtime API, and a writable output directory."""
        workspace = Path(workspace)
        project = workspace/project_directory(workspace_path)
        environment = python_environment()
        command = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL']
        for path in SYSTEM:
            if os.path.isdir(path):
                command += ['--ro-bind', path, path]
        for path in LINKS:
            if os.path.islink(path):
                command += ['--symlink', os.readlink(path), path]
            elif os.path.isdir(path):
                command += ['--ro-bind', path, path]
        command += ['--proc', '/proc', '--dev', '/dev',
                    '--size', str(self.limits.scratch_mb*1024*1024), '--tmpfs', '/tmp',
                    '--ro-bind', str(environment), str(environment),
                    '--ro-bind', str(layout), str(workspace),
                    '--bind', str(writable), str(workspace/'output'),
                    '--ro-bind', str(bundle), str(project),
                    '--ro-bind', str(sdk), str(workspace/'api/runtime.py')]
        if frames is not None:
            command += ['--ro-bind', str(frames), str(workspace/frame_path(frames))]
        variables = {'HOME': str(workspace), 'PATH': f'{environment}/bin:/usr/bin:/bin',
                     'PYTHONPATH': f'{workspace}:{project}', 'PYTHONDONTWRITEBYTECODE': '1',
                     'PYTHONNOUSERSITE': '1', 'PIP_NO_INDEX': '1', 'HF_HUB_OFFLINE': '1',
                     'WANDB_MODE': 'disabled', 'LANG': 'C.UTF-8', **thread_variables(self.limits.cpus)}
        command += ['--chdir', str(workspace), '--clearenv']
        for name, value in variables.items():
            command += ['--setenv', name, value]
        return command + ['--', str(environment/'bin/python'), '-u', str(workspace/'api/runtime.py'), mode,
                          str(project), entrypoint]

    def limit(self):
        """preexec_fn: the sandbox's memory, file and core limits (inherited by everything in it)."""
        memory = self.limits.memory_mb*1024*1024
        resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))
        resource.setrlimit(resource.RLIMIT_FSIZE, (self.limits.artifact_bytes, self.limits.artifact_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (OPEN_FILES, OPEN_FILES))

    def preflight(self):
        binary = shutil.which('bwrap')
        if not binary:
            raise RuntimeError('bubblewrap (bwrap) is required to run bundles')
        help_text = subprocess.run([binary, '--help'], capture_output=True, text=True,
                                   check=True, timeout=10).stdout
        if '--size' not in help_text.split():
            raise RuntimeError(f'bubblewrap at {binary} does not support --size, required to limit '
                               'bundle scratch space; put a compatible bwrap on PATH')
        if not (python_environment()/'bin/python').is_file():
            raise RuntimeError(f'No Python in {python_environment()}')

    def run(self, bundle, *, mode, entrypoint, gateway=None, token=None, output, workspace, name=None,
            frames=None, wall_seconds=None, workspace_path='code/controller'):
        if mode not in {"formal", "rehearsal"} or entrypoint != "controller.py":
            raise ValueError("Unsupported execution mode/entrypoint")
        project = project_directory(workspace_path)
        workspace = Path(workspace).resolve()
        if frames is not None:
            frame_path(frames)
        output.mkdir(mode=0o700, exist_ok=True)
        # A private immutable runtime snapshot, never a mutable skill or developer source path.
        sdk = output / "runtime.py"
        with sdk.open("xb") as stream:
            stream.write(RUNTIME_SOURCE.read_bytes())
        sdk.chmod(0o444)
        # Empty read-only workspace scaffold, never the agent's live filesystem.
        layout = output / 'workspace'
        (layout / project).mkdir(parents=True, mode=0o755)
        (layout / 'output').mkdir(mode=0o755)
        (layout / 'api').mkdir(mode=0o755)
        (layout / 'api/runtime.py').touch(mode=0o444)
        if frames is not None:
            (layout / frame_path(frames)).mkdir(parents=True)
        writable = output / "writable"
        writable.mkdir(mode=0o700)
        end_at = wall_end(self.limits.wall_seconds, wall_seconds)
        output_error = None
        process = None
        reason = "exit"
        returncode = None
        finalizing = False
        execution_started = None
        logs = {"stdout": bytearray(), "stderr": bytearray()}
        seen = {"stdout": 0, "stderr": 0}
        def log(stream, data):
            seen[stream] += len(data)
            logs[stream].extend(data[:max(0, 1024 * 1024 - len(logs[stream]))])
        try:
            remaining = time_left(end_at)
            if remaining is not None and remaining <= 0:
                raise ExecutionDeadline()
            with deadline(remaining):
                execution_started = time.monotonic()
                process = subprocess.Popen(self.command(bundle, mode, entrypoint, workspace=workspace, layout=layout,
                                                        writable=writable, sdk=sdk, frames=frames,
                                                        workspace_path=workspace_path),
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           start_new_session=True, preexec_fn=self.limit)
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ, "mcp")
                selector.register(process.stderr, selectors.EVENT_READ, "log")
                pending = bytearray()
                try:
                    while selector.get_map():
                        for key, _ in selector.select(timeout=1):
                            block = os.read(key.fileobj.fileno(), 65536)
                            if not block:
                                selector.unregister(key.fileobj)
                                continue
                            if key.data == "log":
                                log("stderr", block)
                                continue
                            pending.extend(block)
                            if len(pending) > MAX_REQUEST_BYTES:
                                raise ValueError("Controller request exceeds wire limit")
                            while b"\n" in pending:
                                line, _, tail = pending.partition(b"\n")
                                pending = bytearray(tail)
                                request = json.loads(line)
                                if request.get("method") == "logs/write":
                                    params = request.get("params", {})
                                    if set(params) != {"text"} or not isinstance(params["text"], str):
                                        raise ValueError("Invalid log event")
                                    log("stdout", params["text"].encode())
                                    continue
                                elif finalizing:
                                    response = {'jsonrpc': '2.0', 'id': request.get('id'),
                                                'error': {'code': -32000, 'message': 'Episode ended; artifact cleanup only'}}
                                else:
                                    response = gateway.handle(token, request)
                                if response is not None:
                                    process.stdin.write(json.dumps(response, allow_nan=False).encode() + b"\n")
                                    process.stdin.flush()
                                if gateway is not None and gateway.terminal and not finalizing:
                                    finalizing = True
                                    reason = "episode_ended"
                                    finalize_within(self.finalization_seconds)
                        # Each file is capped at the artifact limit (RLIMIT_FSIZE) and the snapshot
                        # keeps at most that much; this only stops runaway output filling the disk.
                        if output_bytes(writable, 2*self.limits.artifact_bytes) > 2*self.limits.artifact_bytes:
                            raise ValueError(f"Controller output exceeds twice its {self.limits.artifact_bytes}-byte limit")
                    returncode = process.wait(timeout=5)
                finally:
                    selector.close()
        except ExecutionDeadline:
            reason = "finalization_timeout" if finalizing else "timeout"
        finally:
            if execution_started is not None:
                self.execution_seconds = time.monotonic() - execution_started
            if process:
                # The sandbox's own process tree dies with it (--die-with-parent, its PID namespace).
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
                # Drain error output already emitted before a terminal action.
                os.set_blocking(process.stderr.fileno(), False)
                while True:
                    try:
                        block = os.read(process.stderr.fileno(), 65536)
                    except BlockingIOError:
                        break
                    if not block:
                        break
                    log("stderr", block)
                for pipe in (process.stdin, process.stdout, process.stderr):
                    pipe.close()
            output.mkdir(mode=0o700, exist_ok=True)
            for stream, data in logs.items():
                (output / f"{stream}.log").write_bytes(data)
            log_details = {stream: {"bytes_received": seen[stream], "bytes_saved": len(data),
                                   "truncated": seen[stream] > len(data)} for stream, data in logs.items()}
            (output / "logs.json").write_text(json.dumps(log_details))
            # The writer is dead before inspecting untrusted output names. Copy
            # regular files only; never execute/deserialise submitted artifacts.
            try:
                snapshot(writable, output / "files", max_bytes=self.limits.artifact_bytes)
            except Exception as exc:
                output_error = f"{type(exc).__name__}: {public_failure(exc)['reason']}"
            finally:
                shutil.rmtree(writable, ignore_errors=True)
        return {"reason": reason, "returncode": returncode, "logs": log_details,
                "simulator_exited": gateway and gateway.simulator_exited, "output_error": output_error}


def frame_path(frames):
    """Where a run's frames appear in its workspace (runtime/autonomous_controller/runs/<run>/frames)."""
    frames = Path(frames)
    if (not frames.is_absolute() or frames.is_symlink() or not frames.is_dir()
            or frames.name != 'frames' or frames.parent.parent.name != 'runs'
            or not re.fullmatch(r'(?:rehearsal|formal)-[a-f0-9]{32}', frames.parent.name)):
        raise ValueError('Invalid published frame directory')
    return Path('runtime/autonomous_controller/runs', frames.parent.name, 'frames')
