"""Run Claude Code natively as the auto-research agent (``agent_cli: claude``).

Claude Code runs as shipped, with the session's own CLAUDE_CONFIG_DIR and its native login
(login.py); its sandbox confines the agent's commands like Codex's (harness/agent_host.py).
Everything else about a session is shared with the Codex launcher.
"""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from harness import agent_host
from harness.claude_cli.login import PROVIDERS, Credential
from harness.codex_cli.auto_research import (CALL_SECONDS, LOGIN_FILE, PID_FILE, expose_published, native_session,
                                             prepare_workspace, reference_sources, refresh_reviews, start_frontend,
                                             stop_services)
from services.controller.storage import persist

HOME_KEY, HOME_NAME = 'claude_home', 'claude-home'
# Trusted launcher-written configuration a copied session needs (never credentials).
HOME_FILES = ('settings.json', 'mcp.json', '.claude.json')
MCP_SERVER = 'auto_research'
# The agent works offline: its commands have no internet, and neither have its tools.
DENIED_TOOLS = ('WebFetch', 'WebSearch')
# Workflow launches agents independently; ScheduleWakeup creates later autonomous turns.
SUBAGENT_TOOLS = ('Agent', 'Task', 'Workflow', 'ScheduleWakeup')
EFFORT = 'xhigh'  # When a deployment predates agent_effort.
# Compact near this many tokens instead of the model's 1M window: every call re-reads the
# whole context, and an expired cache re-writes all of it.
COMPACT_WINDOW = 250_000
# Claude Code names a project's transcript directory after its working directory.
PROJECT_NAME_LIMIT = 200


def claude_binary(path=None):
    binary = Path(path or shutil.which('claude') or '').resolve()
    if not binary.is_file():
        raise RuntimeError('Claude Code binary is required (install claude on the host)')
    return binary


def claude_workspace(workspace):
    """Claude Code reads CLAUDE.md and .claude/skills; the template keeps AGENTS.md and
    .agents/skills, shared with Codex."""
    (workspace/'CLAUDE.md').write_text('@AGENTS.md\n')
    (workspace/'.claude').mkdir(exist_ok=True)
    (workspace/'.claude/skills').symlink_to('../.agents/skills', target_is_directory=True)


def preflight_login():
    """claude-login: before anything starts, check that the experiment token is accepted.
    A network failure is reported and does not block: Claude Code reports any real provider
    error itself."""
    import http.client
    from harness.claude_cli import experiment_login
    try:
        return experiment_login.check()
    except (OSError, http.client.HTTPException) as exc:
        print(f'Could not verify the Claude experiment token ({type(exc).__name__}: {exc}); continuing',
              file=sys.stderr)
        return None


def deploy(config, *, provider='openrouter', key_file=None, model=None, claude=None, base_url=None):
    """Prepare a new session workspace for Claude Code without starting a model or simulator."""
    binary = claude_binary(claude)
    root = config.root
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (root/'state.json').exists() or (root/'agent-workspace').exists():
        raise FileExistsError('Use a fresh auto-research trial root')
    identity = None
    if provider == 'claude-login' and key_file is None:
        # The dedicated experiment token (never the personal ~/.claude login), read at each launch.
        from harness.claude_cli import experiment_login
        experiment_login.status()  # The token exists and is usable.
        key_file, identity = experiment_login.token_file(), experiment_login.identity()
    Credential(provider, key_file=key_file, base_url=base_url)  # Validate before provisioning.
    prepared = prepare_workspace(config, HOME_NAME)
    persist(root/LOGIN_FILE, {
        'agent_cli': 'claude', 'provider': provider, 'model': model or PROVIDERS[provider]['model'],
        'key_file': str(key_file) if key_file else None,
        **({'base_url': base_url} if base_url else {}),
        # Identity only (no token): which dedicated account this session bills.
        'account': {k: identity.get(k) for k in ('email', 'organization')} if identity else None})
    claude_workspace(prepared['workspace'])
    deployed = {'workspace': prepared['workspace'], 'claude_home': prepared['home'], 'claude': binary,
                'configuration': prepared['configuration'], 'login': root/LOGIN_FILE,
                'subagents': subagent_cap(config), 'resume_session': None}
    write_home(config, deployed)
    if config.adaptation_source:
        from services.adaptation import import_source
        deployed['resume_session'] = import_source(config, deployed['workspace'], deployed['claude_home'])
    return deployed


def project_name(workspace):
    """The transcript directory Claude Code uses for a working directory."""
    name = re.sub(r'[^A-Za-z0-9]', '-', str(Path(workspace).resolve()))
    if len(name) > PROJECT_NAME_LIMIT:
        raise ValueError(f'Workspace path too long for Claude Code transcripts: {workspace}')
    return name


def relocate(home, workspace):
    """Move conversations recorded under another working directory (an adapted or copied
    session) to this workspace's, where Claude Code resumes them."""
    projects = Path(home, 'projects')
    target = projects/project_name(workspace)
    for directory in sorted(projects.glob('*')) if projects.is_dir() else ():
        if directory == target or not directory.is_dir():
            continue
        target.mkdir(parents=True, exist_ok=True)
        for entry in directory.iterdir():
            if not (target/entry.name).exists():
                entry.rename(target/entry.name)
        if not any(directory.iterdir()):
            directory.rmdir()


def transcripts(home):
    """Top-level conversations, oldest first (subagent transcripts live below them)."""
    return sorted(Path(home, 'projects').glob('*/*.jsonl'), key=lambda p: p.stat().st_mtime)


def latest_claude_session(home):
    """Id of the newest top-level Claude Code conversation."""
    found = transcripts(home)
    if not found:
        raise RuntimeError('No Claude Code conversation to resume in '+str(home))
    return found[-1].stem


latest_session = latest_claude_session


def last_error(home):
    """The newest API error Claude Code recorded in its newest conversation, whole, or None."""
    found = transcripts(home)
    if not found:
        return None
    error = None
    with found[-1].open(errors='replace') as stream:
        for line in stream:
            if '"isApiErrorMessage":true' not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            content = (row.get('message') or {}).get('content')
            text = ' '.join(block.get('text', '') for block in content if isinstance(block, dict)) \
                if isinstance(content, list) else str(content or '')
            error = text or json.dumps({k: row.get(k) for k in ('error', 'apiErrorStatus')})
    return error


def session_files(home, session_id):
    """Transcript files that resume a Claude conversation in a fresh home, relative to it."""
    for transcript in Path(home, 'projects').glob(f'*/{session_id}.jsonl'):
        return [transcript.relative_to(home)]
    raise RuntimeError(f'Claude conversation {session_id} not found in {home}')


def redeploy(config, claude=None):
    """Reattach an existing Claude Code session workspace for a resumed launch."""
    root = config.root
    native_session(root, HOME_NAME)
    deployed = {'workspace': root/'agent-workspace', 'claude_home': root/HOME_NAME, 'claude': claude_binary(claude),
                'configuration': root/'configuration.json', 'login': root/LOGIN_FILE,
                'subagents': subagent_cap(config)}
    expose_published(root)
    relocate(deployed['claude_home'], deployed['workspace'])
    deployed['resume_session'] = latest_claude_session(deployed['claude_home'])
    refresh_reviews(config, deployed['workspace'])
    return deployed


def subagent_cap(config):
    from services.mcp_contract import Contract
    return Contract.from_config(config).subagent_cap


def readable(config, deployed):
    """Paths outside the workspace the agent reads: its run tree, or (agent formal) its episode's
    frames, and the benchmark's reference data."""
    runs = [Path(deployed['frames'])] if 'frames' in deployed else [config.root/'published']
    return runs + [source for source, _ in reference_sources(config)]


def protected(workspace):
    """Workspace files that configure Claude Code itself: settings there could lift the sandbox."""
    return [workspace/'.claude', workspace/'.mcp.json']


def settings(config, deployed, socket):
    """Claude Code's user settings for the session: its own defaults, plus the sandbox that
    confines the agent's commands (writes only in the workspace and the robot socket's
    directory, no internet, the repository, sessions and host sockets unreadable) and the
    permission rules that keep its file tools in the workspace and the run tree."""
    workspace = Path(deployed['workspace']).resolve()
    sockets = Path(socket).parent.resolve()
    reads = [path.resolve() for path in readable(config, deployed)]
    # Claude Code checks the requested symlink spelling as well as its real target.
    read_paths = [*reads, *(workspace / relative for _, relative in reference_sources(config))]
    guarded = [str(path) for path in protected(workspace)]
    key_file = login(deployed).get('key_file')
    credentials = [str(Path(key_file).resolve())] if key_file else []
    rule = lambda tool, path: f'{tool}(/{path})'  # Absolute paths are written //like/this.
    return {
        'permissions': {
            # File edits in the workspace need no approval; sandboxed commands neither
            # (autoAllowBashIfSandboxed). Anything else outside the workspace is not allowed.
            'defaultMode': 'acceptEdits',
            'allow': [f'mcp__{MCP_SERVER}', *(rule('Read', f'{path}/**') for path in read_paths)],
            'deny': [*DENIED_TOOLS, *(rule('Read', path) for path in credentials),
                     *(rule('Edit', path) for path in guarded),
                     *(rule('Edit', f'{path}/**') for path in guarded)]},
        'sandbox': {
            'enabled': True, 'failIfUnavailable': True, 'autoAllowBashIfSandboxed': True,
            'allowUnsandboxedCommands': False,
            'filesystem': {'allowWrite': [str(workspace), str(sockets)], 'denyWrite': guarded,
                           'denyRead': [*(str(path) for path in agent_host.hidden_paths(config.root)),
                                        *credentials],
                           'allowRead': [str(workspace), str(sockets), *map(str, reads),
                                         *map(str, agent_host.reexposed_paths())]},
            # No internet (no domain is allowed); the robot socket is a Unix socket.
            'network': {'allowedDomains': [], 'allowAllUnixSockets': True, 'allowLocalBinding': False}},
        'cleanupPeriodDays': 3650}


def write_home(config, deployed, socket=None):
    """Write the session's Claude Code home for this launch: settings, robot tools, trust."""
    home, workspace = Path(deployed['claude_home']), Path(deployed['workspace']).resolve()
    socket = socket or agent_host.socket_path(config.root)
    (home/'settings.json').write_text(json.dumps(settings(config, deployed, socket), indent=2))
    (home/'mcp.json').write_text(json.dumps({'mcpServers': {MCP_SERVER: {
        'type': 'stdio', 'command': str(agent_host.agent_python()),
        'args': ['-I', str(agent_host.BRIDGE), str(socket)]}}}))
    state = home/'.claude.json'  # Claude Code's own state: keep it, add onboarding and trust.
    value = json.loads(state.read_text()) if state.is_file() else {}
    value['hasCompletedOnboarding'] = True
    value.setdefault('projects', {}).setdefault(str(workspace), {}).update(
        hasTrustDialogAccepted=True, hasCompletedProjectOnboarding=True)
    state.write_text(json.dumps(value))


def login(deployed):
    value = json.loads(deployed['login'].read_text())
    if value.get('credentials_file'):
        raise RuntimeError('This session was prepared with the personal Claude login, which experiments no '
                           'longer use; prepare a new session (claude-login uses the experiment token)')
    return value


def credential(deployed):
    value = login(deployed)
    return Credential(value['provider'], key_file=value['key_file'], base_url=value.get('base_url'))


def agent_command(config, deployed, *, prompt, interactive=False, json_output=False):
    """The native Claude Code command line for one run of the agent."""
    denied = DENIED_TOOLS
    if deployed.get('subagents', 1) == 0:
        # Claude Code has no concurrency setting; the cap is stated in the workspace and
        # measured afterwards, but a zero cap removes the subagent tool entirely.
        denied += SUBAGENT_TOOLS
    home = Path(deployed['claude_home'])
    command = [deployed['claude'], '--model', deployed.get('model') or login(deployed)['model'],
               '--effort', getattr(config, 'agent_effort', None) or EFFORT,
               '--mcp-config', home/'mcp.json', '--strict-mcp-config', '--disallowedTools', *denied]
    if deployed.get('resume_session'):
        command += ['--resume', deployed['resume_session']]  # Continue the session's own conversation.
    if not interactive:
        command = [command[0], '-p', *command[1:], '--output-format', 'stream-json' if json_output else 'text',
                   '--verbose']
    return [str(part) for part in (*command, '--', prompt)]


def agent_environment(config, deployed, socket):
    """Claude Code's process environment: its session home, its login, the robot socket and the
    agent's Python."""
    workspace = Path(deployed['workspace']).resolve()
    home = Path(deployed['claude_home']).resolve()
    # HOME must differ from the workspace: Claude Code skips project skills (.claude/skills)
    # when the working directory is the home directory.
    environment = agent_host.environment(home=home, workspace=workspace, socket=socket,
                                         cpus=config.training.cpus, binary=deployed['claude'])
    timeout_ms = str((CALL_SECONDS+300)*1000)
    environment.update({
        'CLAUDE_CONFIG_DIR': str(home),
        **credential(deployed).environment(model=deployed.get('model') or login(deployed)['model']),
        # One robot tool call (an episode, a rehearsal) may block for days; so may a script using them.
        'MCP_TOOL_TIMEOUT': timeout_ms, 'BASH_MAX_TIMEOUT_MS': str(CALL_SECONDS*1000),
        # The session home is no installation to update.
        'DISABLE_AUTOUPDATER': '1',
        # Print mode otherwise stops background subagents 600 s after the main turn ends.
        'CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS': '0',
        'CLAUDE_CODE_AUTO_COMPACT_WINDOW': str(COMPACT_WINDOW),
        # One-hour prompt cache: agents often wait 5-30 minutes on rehearsals and episodes,
        # and a lapsed 5-minute cache re-writes the whole context.
        'ENABLE_PROMPT_CACHING_1H': '1', 'CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL': '1h'})
    if deployed.get('model_relay'):
        relay = deployed['model_relay']
        environment.update(ANTHROPIC_BASE_URL=relay.base_url, ANTHROPIC_AUTH_TOKEN=relay.token)
    return environment


USAGE_PROMPT = ("The model usage limit interrupted this session; it has now been resumed and any background "
                "workers were stopped. Call status, reread memory/PROGRESS.md, and continue from the live state.")
MAX_USAGE_WAITS = 10


def limited_until(deployed, probe=None):
    """claude-login: when the experiment account's usage window resets, if a one-token probe of its
    rate-limit headers finds it exhausted; None otherwise."""
    value = login(deployed)
    if value.get('provider') != 'claude-login':
        return None
    from harness.claude_cli import experiment_login
    return experiment_login.usage_reset((probe or experiment_login.usage)(Path(value['key_file'])))


def until_usage_resets(run, *, deployed, prompt, interactive, on_wait=None, sleep=time.sleep, probe=None):
    """Run the agent. If the experiment account ran out of usage, wait for the reset (the frontend
    and live episodes stay up), then continue the agent's own conversation."""
    waited, waits = 0.0, 0
    while True:
        returncode = run(prompt)
        if interactive or waits >= MAX_USAGE_WAITS:
            return returncode
        until = limited_until(deployed, probe)
        if until is None:
            return returncode
        pause = max(0.0, until+60-time.time())
        if on_wait and pause:
            on_wait(until, waited)
        sleep(pause)
        waited, waits = waited+pause, waits+1
        if on_wait:
            on_wait(None, waited)
        deployed['resume_session'] = latest_claude_session(deployed['claude_home'])
        prompt = USAGE_PROMPT


def launch(config, deployed, *, gemini_key_file=None, prompt, interactive=False, capture=False, on_wait=None,
           agent_runs=None):
    """The frontend, then native Claude Code in the workspace.

    The frontend (the sole owner of the session's episodes) stays up for every agent run of this
    launch: `agent_runs(run)` may call `run(prompt)` again after a run ends (a finished turn, an
    API error), and the open episodes keep running in between, as they do over usage waits."""
    if os.getuid() == 0:
        raise RuntimeError('Run the launcher as a non-root operator')
    credential(deployed)  # Fail before starting anything.
    socket = agent_host.socket_path(config.root)
    write_home(config, deployed, socket)
    relocate(deployed['claude_home'], deployed['workspace'])
    server = agent = None
    value = login(deployed)
    relay_context = nullcontext(None)
    if value['provider'] == 'custom':
        from harness.claude_cli.messages_relay import serve
        relay_context = serve(credential=credential(deployed), model=value['model'],
                              effort=config.agent_effort, audit=config.root/'claude-request-audit.jsonl')
    with relay_context as relay, (config.root/'frontend.log').open('ab') as log:
        if relay is not None:
            deployed = {**deployed, 'model_relay': relay}
        try:
            server = start_frontend(config, deployed, socket, log, key_file=gemini_key_file)

            def run(prompt, **streams):
                nonlocal agent
                command = agent_command(config, deployed, prompt=prompt, interactive=interactive,
                                        json_output=capture)
                agent = agent_host.start(command, cwd=deployed['workspace'], interactive=interactive,
                                         env=agent_environment(config, deployed, socket),
                                         stdin=None if interactive else subprocess.DEVNULL, **streams)
                agent_host.record(agent, config.root/PID_FILE)
                try:
                    stdout, stderr = agent.communicate()
                finally:
                    agent_host.stop(agent)
                    (config.root/PID_FILE).unlink(missing_ok=True)
                return subprocess.CompletedProcess(command, agent.returncode, stdout, stderr)
            if capture:
                return run(prompt, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            def waited(prompt):
                return until_usage_resets(lambda prompt: run(prompt).returncode, deployed=deployed,
                                          prompt=prompt, interactive=interactive, on_wait=on_wait)
            returncode = agent_runs(waited) if agent_runs else waited(prompt)
            return subprocess.CompletedProcess(agent.args, returncode, None, None)
        finally:
            try:
                stop_services(agent, server)
            finally:
                (config.root/PID_FILE).unlink(missing_ok=True)
