"""Native DeepSeek Harness headless driver over the shared RAX session lifecycle."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from harness import agent_host
from harness.cli_runtime import (CALL_SECONDS, LOGIN_FILE, PID_FILE, expose_published,
    native_session, prepare_workspace, reference_sources, refresh_reviews, start_frontend, stop_services)
from harness.deepseek_cli.login import Credential, MODEL
from harness.deepseek_cli.installation import VERSION
from services.controller.storage import persist

HOME_KEY, HOME_NAME = 'dsh_home', 'dsh-home'
# No credential store or private launcher configuration enters frozen packages.
HOME_FILES = ()
HERE = Path(__file__).resolve().parent


def dsh_binary(path=None):
    default = agent_host.PROJECT/f'runtime/agent-backends/deepseek-harness/tools/dsh-{VERSION}/node_modules/.bin/dsh'
    binary = Path(path or os.environ.get('RAX_DSH_BINARY') or shutil.which('dsh') or default).resolve()
    if not binary.is_file():
        raise RuntimeError('DeepSeek Harness is required; pass --agent-binary')
    return binary


def installation(binary):
    for parent in Path(binary).parents:
        if (parent/'node_modules/@deepseek-ai/dsh/package.json').is_file():
            return parent
    raise ValueError('DSH must come from a pinned npm installation')


def credential(deployed):
    value = json.loads(Path(deployed['login']).read_text())
    return Credential(value['provider'], key_file=value.get('key_file'),
                      auth_home=value.get('auth_home'), base_url=value.get('base_url'))


def deploy(config, *, binary=None, model=None, key_file=None, auth_home=None, base_url=None):
    binary = dsh_binary(binary)
    login = Credential(config.agent_provider, key_file=key_file, auth_home=auth_home, base_url=base_url)
    prepared = prepare_workspace(config, HOME_NAME)
    model = model or (MODEL if config.agent_provider == 'openrouter' else 'deepseek-flash')
    persist(config.root/LOGIN_FILE, {'agent_cli': 'dsh', 'model': model, **login.record(),
                                  'binary': str(binary), 'node': shutil.which('node')})
    deployed = {'workspace': prepared['workspace'], HOME_KEY: prepared['home'], 'dsh': binary,
                'configuration': prepared['configuration'], 'login': config.root/LOGIN_FILE,
                'resume_session': None, 'subagents': 0}
    write_home(config, deployed)
    return deployed


def redeploy(config, *, binary=None):
    native_session(config.root, HOME_NAME)
    value = json.loads((config.root/LOGIN_FILE).read_text())
    deployed = {'workspace': config.root/'agent-workspace', HOME_KEY: config.root/HOME_NAME,
                'dsh': dsh_binary(binary or value['binary']), 'configuration': config.root/'configuration.json',
                'login': config.root/LOGIN_FILE, 'subagents': 0,
                'resume_session': latest_session(config.root/HOME_NAME)}
    expose_published(config.root)
    refresh_reviews(config, deployed['workspace'])
    credential(deployed)
    return deployed


def boundary(config, deployed, socket):
    workspace = Path(deployed['workspace']).resolve()
    login = json.loads(Path(deployed['login']).read_text())
    credentials = [Path(login[name]).resolve() for name in ('key_file', 'auth_home') if login.get(name)]
    exposed = [config.root/'published', *agent_host.reexposed_paths(),
               *(source for source, _ in reference_sources(config))]
    return {'bwrap': shutil.which('bwrap') or '/usr/bin/bwrap', 'workspace': str(workspace),
            'socket_directory': str(Path(socket).parent.resolve()),
            'hidden': [str(p) for p in (*agent_host.hidden_paths(config.root), *credentials)],
            'reexposed': [str(p.resolve()) for p in exposed],
            'file_read_roots': [str(workspace), *[str(p.resolve()) for p in exposed],
                                *[str(Path(p).resolve()) for p in ('/usr', '/bin', '/lib', '/lib64', '/etc')]]}


def patches(config, deployed, socket):
    value = json.loads(Path(deployed['login']).read_text())
    route, provider = credential(deployed).patches(value['model'], config.agent_effort)
    home = Path(deployed[HOME_KEY])
    rows = [*provider,
        {'id': 'agent-default-model', 'config': {'provider': route, 'model': value['model']}},
        {'id': 'approval', 'config': {'policy': 'never'}},
        {'id': 'permission', 'config': {'defaultPreset': 'rax',
            'presets': {'rax': {'sandbox': 'workspace-write', 'approval': 'never'}}}},
        {'id': 'sandbox-policy', 'config': {'mode': 'workspace-write', 'workspaceRoot': str(deployed['workspace'])}},
        {'id': 'sandbox', 'config': {'runnerCommand': [str(agent_host.agent_python()), '-I',
            str(HERE/'runner.py'), str(home/'boundary.json')],
            'runnerFailureSignatures': ['bwrap:', 'RAX_DSH_SANDBOX:']}},
        {'id': 'fs-sandbox', 'name': str(HERE/'filesystem.mjs'), 'config': {'cwd': str(deployed['workspace'])}},
        {'id': 'bash-sandbox', 'config': {'timeoutMs': CALL_SECONDS*1000}},
        {'id': 'compaction-basic', 'config': {'thresholdRatio': 250000/1048576}},
        {'id': 'session-persistence-jsonl', 'config': {'root': str(home/'sessions'), 'compression': 'none'}},
        {'insert': [{'id': 'rax-mcp', 'name': '@deepseek-ai/dsh-mcp-client', 'required': True,
            'config': {'serverName': 'auto_research', 'transport': 'stdio',
                'command': str(agent_host.agent_python()), 'args': ['-I', str(agent_host.BRIDGE), str(socket)],
                'toolCallTimeoutMs': (CALL_SECONDS+300)*1000, 'failOnStartupError': True}}]}]
    # Native web/subagent/background-workflow/interactive tools must not add a
    # second model, network channel, or human approval wait to this experiment.
    disabled = ('tool-web', 'tool-subagent', 'tool-subagent-fork', 'tool-subagent-control',
                'tool-subagent-list-agents', 'workflow-ptc', 'tool-workflow', 'tool-plugin-manager',
                'plugin-manager', 'session-title-llm', 'tool-goal', 'plan-mode')
    rows += [{'id': name, 'disabled': True} for name in disabled]
    return rows


def write_home(config, deployed, socket=None):
    socket = socket or agent_host.socket_path(config.root)
    home = Path(deployed[HOME_KEY])
    profile = home/'profiles/headless'
    profile.mkdir(parents=True, exist_ok=True)
    if not (profile/'package.json').exists():
        persist(profile/'package.json', {'name': 'rax-dsh-headless', 'private': True, 'dependencies': {},
            'dsh': {'profile': {'bundles': ['@deepseek-ai/dsh-base', '@deepseek-ai/dsh-headless']}}})
        (profile/'cordis.yml').write_text('[]\n')
    persist(home/'boundary.json', boundary(config, deployed, socket))
    persist(profile/'cordis.patch.yml', patches(config, deployed, socket))


def agent_command(config, deployed, *, prompt, interactive=False, json_output=False):
    if interactive:
        raise ValueError('DSH currently uses the non-interactive headless driver')
    value = json.loads(Path(deployed['login']).read_text())
    command = [value['node'], str(deployed['dsh']), '--profile', 'headless', '--json']
    if deployed.get('resume_session'):
        command += ['--session-id', deployed['resume_session']]
    return [*command, prompt]


def agent_environment(config, deployed, socket):
    value = json.loads(Path(deployed['login']).read_text())
    env = agent_host.environment(home=deployed['workspace'], workspace=deployed['workspace'],
        socket=socket, cpus=config.training.cpus, binary=value['node'])
    env.update(credential(deployed).environment(), DSH_HOME=str(deployed[HOME_KEY]),
        RAX_DSH_INSTALL=str(installation(deployed['dsh'])), RAX_DSH_BOUNDARY=str(Path(deployed[HOME_KEY])/'boundary.json'))
    env.update(NODE_OPTIONS=f'--import={HERE/"request_audit.mjs"}',
               RAX_DSH_AUDIT=str(Path(deployed[HOME_KEY])/'request-audit.jsonl'))
    return env


def latest_session(home):
    path = Path(home)/'rax-session.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())['session_id']


def session_files(home, session_id):
    root = Path(home)
    # The native store owns encoded IDs and generation names. Preserve durable
    # session files, never auth, caches, operator settings, or process logs.
    return [path.relative_to(root) for path in sorted((root/'sessions').rglob('*'))
            if path.is_file() and not path.is_symlink()]


def last_error(home):
    path = Path(home)/'rax-error.json'
    return json.loads(path.read_text())['error'] if path.exists() else None


def record_subagents(config, deployed):
    persist(config.root/'subagents.json', {'cli': 'dsh', 'cap': 0, 'spawned': 0,
        'enforcement': 'native subagent and workflow tools disabled'})


def capture_event(home, line):
    try:
        event = json.loads(line)
    except ValueError:
        return
    if not isinstance(event, dict):
        return
    if event.get('type') == 'session':
        ident = event.get('sessionId') or event.get('session_id')
        if ident:
            persist(Path(home)/'rax-session.json', {'session_id': ident})
    if event.get('type') == 'error':
        persist(Path(home)/'rax-error.json', {'error': event.get('message', str(event))})
    if event.get('type') == 'status' and event.get('phase') == 'step_end':
        with (Path(home)/'usage.jsonl').open('a') as stream:
            stream.write(json.dumps({'session_id': latest_session(home), **event})+'\n')


def launch(config, deployed, *, gemini_key_file=None, prompt, interactive=False, capture=False,
           agent_runs=None, **unused):
    if capture or interactive:
        raise ValueError('DSH launch requires non-interactive streaming mode')
    credential(deployed)
    socket = agent_host.socket_path(config.root)
    write_home(config, deployed, socket)
    server = agent = None
    home = Path(deployed[HOME_KEY])
    with (config.root/'frontend.log').open('ab') as log:
        try:
            server = start_frontend(config, deployed, socket, log, key_file=gemini_key_file)
            def run(text):
                nonlocal agent
                (home/'rax-error.json').unlink(missing_ok=True)
                command = agent_command(config, deployed, prompt=text)
                agent = agent_host.start(command, cwd=deployed['workspace'],
                    env=agent_environment(config, deployed, socket), stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, text=True)
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
