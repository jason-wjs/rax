"""The agent CLI on the host: its sandbox paths, robot-tool socket, environment and processes.

Codex and Claude Code run natively with their own logins and reach their model providers
directly. Each CLI's own sandbox confines the commands the agent runs: writes only in its
workspace, no internet, no host processes, and the repository, every session's private
files and the host's sockets unreadable. The session's frontend serves the robot tools on
a Unix socket; the CLI reaches it through mcp_bridge.py, the agent's Python through
RAX_MCP_SOCKET (auto_research_agent/api/runtime.py).
"""
from contextlib import contextmanager, suppress
import ctypes
import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
BRIDGE = PROJECT/'harness/mcp_bridge.py'
SOCKET_ENV = 'RAX_MCP_SOCKET'
UID = os.getuid()
SOCKETS = Path(tempfile.gettempdir())/f'rax-{UID}'
# On Linux both CLIs' sandboxes allow every Unix socket or none, and the agent's Python
# needs the robot socket. These host sockets would let a command leave the sandbox
# (docker, containerd) or reach the desktop session, so the sandbox hides them.
HOST_SOCKETS = (Path('/run/docker.sock'), Path('/var/run/docker.sock'), Path('/run/containerd'),
                Path('/run/dbus'), Path(f'/run/user/{UID}'), Path('/tmp/.X11-unix'))
# The operator's credentials (any agent's own login stays readable to its CLI, outside the sandbox):
# among them the logins a Codex account switcher keeps in ~/.codex/accounts and the GitHub CLI's token.
# All of ~/.codex: its logins and keys, provider profiles (which may carry a key), history and transcripts.
HOME = Path.home()
SECRETS = ('.ssh', '.aws', '.config/gcloud', '.netrc', '.git-credentials', '.config/rax', '.cursor', '.config/cursor', '.claude',
           '.claude.json', '.codex', '.config/gh')
# Codex's standalone install under ~/.codex stays readable: Codex re-runs its own binary in the sandbox.
CODEX_INSTALL = '.codex/packages'
# Under host memory pressure the kernel kills the agent's processes before the simulators.
OOM_SCORE_ADJ = 500
FREEZE_SECONDS = 10  # The longest frozen() waits for the agent's processes to stop.
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS')
# The project links that the agent's Python environment lives under (only its envs stay readable).
RUNTIME = 'runtime'
ENVS = 'envs'


def socket_path(root):
    """The session's robot socket, alone in its own directory: the sandbox exposes that
    directory and hides every other session's (Linux socket paths are limited to 107 bytes)."""
    SOCKETS.mkdir(mode=0o700, exist_ok=True)
    directory = SOCKETS/hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:16]
    directory.mkdir(mode=0o700, exist_ok=True)
    return directory/'mcp.sock'


def agent_python():
    """The interpreter on the agent's PATH: this environment by its real path, since the
    repository path that links to it is hidden from the sandbox."""
    return Path(sys.prefix).resolve()/'bin/python'


def host_secrets():
    """The operator's credentials under their home: SECRETS and every other Codex home's login
    (~/.codex-*/auth.json). Few, whole directories: Claude Code's sandbox command grows with every
    path it hides, and fails beyond about a hundred of them."""
    return [*(HOME/name for name in SECRETS), *HOME.glob('.codex-*/auth.json')]


def reexposed_paths():
    """Readable inside hidden directories: the Python environments in the runtime (the agent's
    Python among them) and Codex's own install, which Codex runs inside its sandbox."""
    return [path.resolve() for path in (PROJECT/RUNTIME/ENVS, HOME/CODEX_INSTALL) if _exists(path)]


def _exists(path):
    """Whether a path exists; one this user cannot even stat is as unreadable to the agent."""
    try:
        return path.exists()
    except OSError:
        return False


def hidden_paths(root, *, carved=False, kept=()):
    """What the agent's commands must not read: the repository, the assets, checkouts and
    runtime it links to (except the Python environments), every session's directory (the
    agent's own workspace is re-exposed writable), the other robot sockets, host sockets and
    the operator's credentials.

    carved: for a sandbox (Codex's) that cannot make a path readable again inside a hidden one,
    a hidden directory holding a reexposed_paths() entry, or one of ``kept`` (the session's
    reference data), is replaced by its other entries."""
    paths = [*repositories(), Path(root).resolve().parent, SOCKETS,
             *(link.resolve() for link in sorted(PROJECT.iterdir()) if link.is_symlink())]
    # Resolved (a sandbox mounts over real paths, not links), existing, outermost only.
    resolved = [*dict.fromkeys(path.resolve() for path in (*paths, *HOST_SOCKETS, *host_secrets())
                               if _exists(path))]
    outermost = [path for path in resolved if not any(path != other and path.is_relative_to(other)
                                                      for other in resolved)]
    if not carved:
        return outermost
    kept = [*reexposed_paths(), *(Path(path).resolve() for path in kept)]
    return [part for path in outermost for part in carve(path, kept)]


def carve(path, kept):
    """``path`` to hide, or, when it holds a ``kept`` path, its entries but that one, carved likewise.
    An entry that links outside ``path`` is not part of it and is left out."""
    if path in kept:
        return []
    if not any(other.is_relative_to(path) for other in kept):
        return [path]
    try:
        entries = sorted(path.iterdir())
    except OSError:  # Unlistable: hidden whole, the kept path with it.
        return [path]
    return [part for entry in entries if entry.resolve().is_relative_to(path)
            for part in carve(entry.resolve(), kept)]


def repositories():
    """This checkout and, for a git worktree, the repository it belongs to."""
    paths = [PROJECT]
    marker = PROJECT/'.git'
    if marker.is_file():  # A worktree: "gitdir: <repository>/.git/worktrees/<name>"
        gitdir = Path(marker.read_text().partition(':')[2].strip())
        if '.git' in gitdir.parts:
            paths.append(Path(*gitdir.parts[:gitdir.parts.index('.git')]))
    return paths


def thread_variables(cpus):
    """Size math-library thread pools to the agent's share of the host, not every core."""
    threads = str(max(1, math.ceil(float(cpus))))
    return {name: threads for name in THREAD_VARIABLES}


def environment(*, home, workspace, socket, cpus, binary=None):
    """The agent CLI's environment: nothing inherited but terminal and locale basics."""
    variables = {k: os.environ[k] for k in ('TERM', 'LANG', 'LC_ALL', 'COLORTERM', 'USER', 'LOGNAME', 'TZ')
                 if k in os.environ}
    path = [str(agent_python().parent), *([str(Path(binary).parent)] if binary else []),
            '/usr/local/bin', '/usr/bin', '/bin']
    variables.update({'HOME': str(home), 'PATH': ':'.join(dict.fromkeys(path)), SOCKET_ENV: str(socket),
                      'PYTHONPATH': str(workspace), 'PYTHONDONTWRITEBYTECODE': '1', 'PIP_NO_INDEX': '1',
                      'HF_HUB_OFFLINE': '1', **thread_variables(cpus)})
    return variables


def deprioritize():
    """preexec_fn for agent processes; inherited by everything the agent starts."""
    with open('/proc/self/oom_score_adj', 'w') as stream:
        stream.write(str(OOM_SCORE_ADJ))


def start(command, *, cwd, env, interactive=False, **streams):
    """Start the agent CLI under supervise() (this file, run isolated from the workspace's
    Python path). A non-interactive agent gets its own session."""
    wrapper = [sys.executable, '-I', str(Path(__file__).resolve()), '--', *map(str, command)]
    return subprocess.Popen(wrapper, cwd=cwd, env=env, preexec_fn=deprioritize,
                            start_new_session=not interactive, **streams)


PR_SET_CHILD_SUBREAPER = 36


def supervise(command):
    """Run the agent CLI and, once it exits, end everything it left running. This process
    adopts the CLI's orphaned descendants (a child subreaper), so a background command that
    outlived the CLI is still found. Returns the CLI's exit status."""
    if ctypes.CDLL(None, use_errno=True).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'Cannot adopt the agent CLI\'s processes')
    child = subprocess.Popen(command)
    for number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, lambda number, _: child.send_signal(number))
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # A terminal's Ctrl-C is the CLI's own.
    returncode = child.wait()
    for _ in range(200):
        left = descendants(os.getpid())
        if not left:
            break
        for pid in left:
            with suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break
        time.sleep(.05)
    return returncode if returncode >= 0 else 128-returncode


def descendants(pid):
    """Every live process below `pid` (the CLI's sandboxed commands included)."""
    found, frontier = [], [pid]
    while frontier:
        parent = frontier.pop()
        try:
            tasks = os.listdir(f'/proc/{parent}/task')
        except OSError:
            continue
        for task in tasks:
            try:
                children = Path(f'/proc/{parent}/task/{task}/children').read_text().split()
            except OSError:
                continue
            for child in map(int, children):
                if child not in found:
                    found.append(child)
                    frontier.append(child)
    return found


def signal_tree(pid, number):
    """Signal a process and its descendants, walking again until no new one appears."""
    signalled = []
    for _ in range(100):
        new = [p for p in (pid, *descendants(pid)) if p not in signalled]
        if not new:
            break
        for member in new:
            with suppress(ProcessLookupError, PermissionError):
                os.kill(member, number)
            signalled.append(member)
    return signalled


def start_time(pid):
    """The kernel's start time of `pid`, which tells it apart from a later process reusing the id."""
    stat = Path(f'/proc/{pid}/stat').read_text()
    return int(stat[stat.rindex(')')+2:].split()[19])


def record(process, path):
    """Write the agent's pid file for the frontend (frozen())."""
    path.write_text(f'{process.pid} {start_time(process.pid)}\n')


def recorded(path):
    """The live agent process named by a pid file, or None."""
    try:
        pid, started = map(int, Path(path).read_text().split())
        return pid if start_time(pid) == started else None
    except (OSError, ValueError):
        return None


def stopped(pid):
    """True once the kernel has stopped `pid` (or it is gone)."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
    except OSError:
        return True
    return stat[stat.rindex(')')+2] in 'TtZX'


@contextmanager
def frozen(pid_file):
    """Stop the agent CLI and everything it runs for the block: a background writer must not
    change a submission while it is copied or executed. SIGSTOP reaches the parent first,
    so nothing forks past the walk."""
    pid = recorded(pid_file) if pid_file else None
    members = signal_tree(pid, signal.SIGSTOP) if pid else []
    # kill() returns before the kernel stops a process: wait until every one has (a process in
    # uninterruptible I/O stops when the I/O ends).
    deadline = time.monotonic() + FREEZE_SECONDS
    while not all(map(stopped, members)) and time.monotonic() < deadline:
        time.sleep(.005)
    try:
        yield
    finally:
        for member in members:
            with suppress(ProcessLookupError, PermissionError):
                os.kill(member, signal.SIGCONT)


def stop(process, grace=30):
    """End the agent CLI started by start(); its supervisor ends what the CLI left running."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        signal_tree(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


if __name__ == '__main__':
    raise SystemExit(supervise(sys.argv[sys.argv.index('--')+1:]))
