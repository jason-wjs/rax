"""Native Cursor CLI lifecycle; the vendor owns the agent loop and account routing."""
import json
import os
from pathlib import Path
import shutil
import shlex
import stat
import subprocess
import time

from harness import agent_host
from harness.cli_runtime import (LOGIN_FILE, PID_FILE, expose_published, native_session,
    prepare_workspace, reference_sources, refresh_reviews, start_frontend, stop_services)
from services.controller.storage import persist

HOME_KEY, HOME_NAME, HOME_FILES = 'cursor_home', 'cursor-home', ()
MODEL = 'grok-4.7[context=500k,reasoning_effort=xhigh,fast=true]'


def cursor_binary(path=None):
    path = path or os.environ.get('RAX_CURSOR_BINARY') or shutil.which('agent')
    if not path or not Path(path).is_file():
        raise RuntimeError('Cursor CLI is required; pass --agent-binary')
    return Path(path).resolve()


def account_home(path=None):
    home = Path(path or agent_host.HOME/'.config/cursor').resolve()
    if home.name != 'cursor' or not (home/'auth.json').is_file():
        raise ValueError('Cursor account login needs an authenticated XDG cursor directory with auth.json')
    info = (home/'auth.json').lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('Cursor auth.json must be private to its owner')
    return home


def deploy(config, *, binary=None, model=None, key_file=None, auth_home=None, base_url=None):
    if base_url:
        raise ValueError('Cursor uses its native account service, not a custom model endpoint')
    binary, auth = cursor_binary(binary), account_home(auth_home)
    prepared = prepare_workspace(config, HOME_NAME)
    version = subprocess.check_output([str(binary), '--version'], text=True, timeout=30).strip()
    persist(config.root/LOGIN_FILE, {'agent_cli': 'cursor', 'provider': 'cursor-account',
        'model': model or MODEL, 'effort': config.agent_effort, 'binary': str(binary),
        'version': version, 'auth_home': str(auth)})
    deployed = {'workspace': prepared['workspace'], HOME_KEY: prepared['home'], 'cursor': binary,
        'configuration': prepared['configuration'], 'login': config.root/LOGIN_FILE,
        'resume_session': None, 'subagents': 0}
    write_home(config, deployed)
    return deployed


def redeploy(config, *, binary=None):
    native_session(config.root, HOME_NAME)
    value = json.loads((config.root/LOGIN_FILE).read_text())
    account_home(value['auth_home'])
    session = latest_session(config.root/HOME_NAME)
    if not session:
        raise RuntimeError('No Cursor conversation to resume; inspect the previous launch error')
    deployed = {'workspace': config.root/'agent-workspace', HOME_KEY: config.root/HOME_NAME,
        'cursor': cursor_binary(binary or value['binary']), 'configuration': config.root/'configuration.json',
        'login': config.root/LOGIN_FILE, 'subagents': 0,
        'resume_session': session}
    expose_published(config.root)
    refresh_reviews(config, deployed['workspace'])
    return deployed


def write_home(config, deployed, socket=None):
    socket = socket or agent_host.socket_path(config.root)
    workspace, home = Path(deployed['workspace']).resolve(), Path(deployed[HOME_KEY]).resolve()
    value = json.loads(Path(deployed['login']).read_text())
    reads = list(dict.fromkeys(str(Path(p).resolve()) for p in (
        config.root/'published', socket.parent, *agent_host.reexposed_paths(),
        *(source for source, _ in reference_sources(config)))))
    private = [str(home), value['auth_home'], *map(str, agent_host.hidden_paths(config.root, carved=True,
        kept=(workspace, config.root/'published', socket.parent,
              *(source for source, _ in reference_sources(config)))))]
    permissions = {
        'allow': ['Shell(*)', f'Read({workspace}/**)', f'Write({workspace}/**)',
                  'Mcp(auto_research:*)', *(f'Read({path}/**)' for path in reads)],
        'deny': ['WebFetch(*)', *(rule for path in private for rule in (f'Read({path})', f'Read({path}/**)')),
                 *(rule for path in (*private, str(workspace/'.cursor'))
                   for rule in (f'Write({path})', f'Write({path}/**)'))]}
    settings_path = home/'cli-config.json'
    previous = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    # Keep the vendor's account identity, which binds its cached feature metadata.
    # Credentials and feature settings are not copied into the generated config.
    auth_info = previous.get('authInfo') or {}
    identity = {key: auth_info[key] for key in ('email', 'displayName', 'teamId', 'teamName',
        'userId', 'authId', 'organizationId', 'activeTeamId') if key in auth_info}
    persist(settings_path, {'version': 1, 'editor': {'vimMode': False},
        **({'authInfo': identity} if identity else {}),
        'approvalMode': 'allowlist', 'autoAcceptWebSearch': False, 'permissions': permissions,
        'sandbox': {'mode': 'enabled', 'readBoundary': 'workspace', 'networkAccess': 'user_config_only'}})
    cursor = workspace/'.cursor'
    cursor.mkdir(exist_ok=True)
    persist(cursor/'cli.json', {'permissions': permissions})
    persist(cursor/'sandbox.json', {'type': 'workspace_readwrite', 'readBoundary': 'workspace',
        'additionalReadPaths': reads, 'additionalReadonlyPaths': [*reads, str(cursor)],
        'networkPolicy': {'default': 'deny', 'allow': [], 'deny': ['*']}})
    persist(cursor/'mcp.json', {'mcpServers': {'auto_research': {
        'command': str(agent_host.agent_python()), 'args': ['-I', str(agent_host.BRIDGE), str(socket)]}}})
    persist(cursor/'hooks.json', {'version': 1, 'hooks': {'subagentStart': [{
        'command': shlex.join([str(agent_host.agent_python()), '-I',
                              str(Path(__file__).with_name('deny_subagents.py'))]),
        'failClosed': True}]}})


def agent_command(config, deployed, *, prompt, interactive=False, json_output=False):
    if interactive:
        raise ValueError('Cursor currently uses the non-interactive CLI driver')
    value = json.loads(Path(deployed['login']).read_text())
    command = [str(deployed['cursor']), '--print', '--output-format', 'stream-json',
        '--sandbox', 'enabled', '--approve-mcps', '--trust', '--workspace', str(deployed['workspace']),
        '--model', value['model']]
    if deployed.get('resume_session'):
        command += ['--resume', deployed['resume_session']]
    return [*command, '--', prompt]


def agent_environment(config, deployed, socket):
    home = Path(deployed[HOME_KEY])
    value = json.loads(Path(deployed['login']).read_text())
    auth = account_home(value['auth_home'])
    env = agent_host.environment(home=home, workspace=deployed['workspace'], socket=socket,
        cpus=config.training.cpus, binary=deployed['cursor'])
    env.update(CURSOR_CONFIG_DIR=str(home), CURSOR_DATA_DIR=str(home/'data'),
               XDG_CONFIG_HOME=str(auth.parent), AGENT_CLI_CREDENTIAL_STORE='file')
    return env


def latest_session(home):
    path = Path(home)/'rax-session.json'
    return json.loads(path.read_text())['session_id'] if path.exists() else None


def session_files(home, session_id):
    home = Path(home)
    return [p.relative_to(home) for folder in sorted((home/'chats').glob('*/*'))
            if folder.name == session_id and not folder.is_symlink()
            for p in sorted(folder.rglob('*')) if p.is_file() and not p.is_symlink()
            and not any(parent.is_symlink() for parent in p.parents if home in parent.parents)]


def last_error(home):
    path = Path(home)/'rax-error.json'
    return json.loads(path.read_text())['error'] if path.exists() else None


def capture_event(home, line):
    try:
        event = json.loads(line)
    except ValueError:
        return
    if not isinstance(event, dict):
        return
    home = Path(home)
    if event.get('type') == 'system' and event.get('subtype') == 'init':
        persist(home/'rax-session.json', {'session_id': event['session_id'], 'model': event.get('model')})
    if event.get('type') == 'error' or event.get('type') == 'result' and event.get('is_error'):
        persist(home/'rax-error.json', {'error': event.get('message') or event.get('result') or event.get('subtype')})
    if event.get('type') == 'result':
        row = {k: event.get(k) for k in ('session_id', 'request_id', 'duration_ms', 'duration_api_ms',
                                        'usage', 'is_error', 'subtype')}
        with (home/'usage.jsonl').open('a') as stream:
            stream.write(json.dumps({'at': time.time(), **row})+'\n')


def record_subagents(config, deployed):
    persist(config.root/'subagents.json', {'cli': 'cursor', 'cap': 0, 'spawned': 0,
        'enforcement': 'fail-closed native subagentStart hook denies spawning'})


def preflight(config, deployed):
    """Check effective native isolation, including server-gated read boundaries."""
    home, workspace = Path(deployed[HOME_KEY]), Path(deployed['workspace'])
    canary = home/'boundary-canary.txt'
    canary.write_text('Private launcher canary, not a task input')
    script = f'''
import os, socket
from pathlib import Path
try: data=Path({str(canary)!r}).read_bytes()
except OSError: pass
else: assert not data, "Private launcher file is readable"
assert not list(Path('/dev').glob('nvidia*')), "GPU exposed"
assert not Path('/proc/{os.getpid()}/cmdline').exists(), "Host process visible"
try: socket.create_connection(('1.1.1.1',443),timeout=1).close()
except OSError: pass
else: raise AssertionError("Internet is reachable")
path=Path({str(workspace/'output/cursor-boundary-ok.txt')!r})
path.write_text('OK')
print('RAX_CURSOR_BOUNDARY_OK')
'''
    try:
        # Cursor's sandbox command joins its arguments into a shell command.
        result = subprocess.run([str(deployed['cursor']), 'sandbox', 'run', '--', '/usr/bin/python3', '-I', '-c', shlex.quote(script)],
            cwd=workspace, env=agent_environment(config, deployed, agent_host.socket_path(config.root)),
            capture_output=True, text=True, timeout=45)
        if result.returncode or 'RAX_CURSOR_BOUNDARY_OK' not in result.stdout.splitlines():
            persist(home/'boundary-check.json', {'passed': False, 'returncode': result.returncode})
            raise RuntimeError('Cursor effective sandbox boundary failed; no model run was started')
        persist(home/'boundary-check.json', {'passed': True, 'checked_at': time.time(),
            'private_read_denied': True, 'no_gpu': True, 'no_host_processes': True, 'no_internet': True})
    finally:
        canary.unlink(missing_ok=True)


def launch(config, deployed, *, gemini_key_file=None, prompt, interactive=False, capture=False,
           agent_runs=None, **unused):
    if interactive or capture:
        raise ValueError('Cursor launch requires non-interactive streaming mode')
    socket = agent_host.socket_path(config.root)
    write_home(config, deployed, socket)
    preflight(config, deployed)
    home = Path(deployed[HOME_KEY])
    server = agent = None
    with (config.root/'frontend.log').open('ab') as log:
        try:
            server = start_frontend(config, deployed, socket, log, key_file=gemini_key_file)
            def run(text):
                nonlocal agent
                (home/'rax-error.json').unlink(missing_ok=True)
                agent = agent_host.start(agent_command(config, deployed, prompt=text),
                    cwd=deployed['workspace'], env=agent_environment(config, deployed, socket),
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                agent_host.record(agent, config.root/PID_FILE)
                try:
                    for line in agent.stdout:
                        capture_event(home, line)
                        print(line, end='', flush=True)
                    return agent.wait()
                finally:
                    agent_host.stop(agent)
                    (config.root/PID_FILE).unlink(missing_ok=True)
            code = agent_runs(run) if agent_runs else run(prompt)
            return subprocess.CompletedProcess(agent.args, code)
        finally:
            stop_services(agent, server)
