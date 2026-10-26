"""Vendor-owned agent loops in offline Docker containers, with host-only credentials."""
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from harness import agent_host
from harness.cli_runtime import CALL_SECONDS, reference_sources
from harness.codex_cli import auto_research as lifecycle, workspace_storage
from harness.isolated_cli.relay import serve
from services.controller.sandbox import DockerSandbox, thread_env

HERE = Path(__file__).resolve().parent
SUPPORT = Path('/opt/rax-cli')


def stage_support(root):
    """Only the public transport helpers enter the container runtime."""
    target = Path(root)/'container-support'
    files = ('container_cli/runner.py', 'isolated_cli/transport.mjs',
             'deepseek_cli/private_provider.mjs', 'deepseek_cli/message_images.mjs', 'mcp_bridge.py')
    for relative in files:
        destination = target/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE.parent/relative, destination)
    return target


class DeepSeekCredential:
    def __init__(self, native, deployed):
        self.native, self.deployed = native, deployed
        self.login = native.credential(deployed)
        self.provider = self.login.provider
        self.endpoint = ('https://openrouter.ai/api/v1' if self.provider == 'openrouter'
                         else 'https://api.deepseek.com/anthropic' if self.provider != 'custom'
                         else self.login.base_url)

    def headers(self):
        if self.provider != 'deepseek-account':
            from harness.claude_cli.login import private_text
            return {'Authorization': 'Bearer '+private_text(self.login.key_file).strip()}
        value = json.loads(Path(self.deployed['login']).read_text())
        result = subprocess.run([value['node'], str(HERE.parent/'deepseek_cli/account_headers.mjs'),
            str(self.native.installation(self.deployed['dsh'])), str(self.login.credentials_file)],
            capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError('DeepSeek account credential resolution failed')
        return json.loads(result.stdout)['headers']


class Adapter:
    def __init__(self, name, native):
        self.name, self.native = name, native
        self.HOME_KEY, self.HOME_NAME, self.HOME_FILES = native.HOME_KEY, native.HOME_NAME, native.HOME_FILES
        self.spec = None if name == 'dsh' else importlib.import_module('harness.'+name+'_cli.spec')

    def __getattr__(self, name):
        return getattr(self.native, name)

    def deploy(self, config, **options):
        deployed = self.native.deploy(config, **options)
        self.write_home(config, deployed)
        return deployed

    def redeploy(self, config, **options):
        deployed = self.native.redeploy(config, **options)
        self.write_home(config, deployed)
        return deployed

    def write_home(self, config, deployed, socket=None):
        socket = socket or agent_host.socket_path(config.root)
        self.native.write_home(config, deployed, socket)
        stage_support(config.root)
        home = Path(deployed[self.HOME_KEY])
        if self.name == 'dsh':
            patch = home/'profiles/headless/cordis.patch.yml'
            rows = json.loads(patch.read_text())
            rows = [row for row in rows if row.get('id') not in ('fs-sandbox', 'sandbox',
                'llm-deepseek', 'llm-deepseek-account', 'credentials', 'authorization', 'deepseek-account')
                and not any(p.get('id') == 'rax-private' for p in row.get('insert', []))]
            # The outer container owns process and filesystem isolation.
            rows += [{'id': 'sandbox', 'config': {'runnerCommand':
                ['/usr/local/bin/python', '-I', str(HERE/'runner.py')],
                'runnerFailureSignatures': ['RAX_DSH_CONTAINER:']}},
                {'id': 'fs-sandbox', 'config': {'cwd': str(deployed['workspace'])}}]
            if config.agent_provider == 'openrouter':
                for row in rows:
                    if row.get('id') == 'llm-pi-ai':
                        row['config']['providers']['rax-model']['baseURL'] = 'http://rax-model.invalid/v1'
            else:
                rows += [{'insert': [{'id': 'rax-private',
                    'name': str(HERE.parent/'deepseek_cli/private_provider.mjs'),
                    'config': {'baseURL': 'http://rax-model.invalid/v1', 'thinking': 'enabled',
                               'reasoningEffort': 'max'}}]}]
                next(row for row in rows if row.get('id') == 'agent-default-model')['config']['provider'] = 'rax-private'
            for row in rows:
                for plugin in row.get('insert', []):
                    if plugin.get('id') == 'rax-mcp':
                        plugin['config']['command'] = '/usr/local/bin/python'
                        plugin['config']['args'] = ['-I', '/tmp/rax-bridge/mcp_bridge.py', str(socket)]
            patch.write_text(json.dumps(rows).replace(str(HERE.parent), str(SUPPORT)))
        else:
            path = home/'mcp.json'
            if path.exists():
                path.write_text(path.read_text().replace(str(agent_host.agent_python()), '/usr/local/bin/python'))

    def agent_environment(self, config, deployed, socket):
        value = json.loads(Path(deployed['login']).read_text())
        home = Path(deployed[self.HOME_KEY])
        variables = {'HOME': str(home), 'PATH': str(Path(value['node']).parent)+':/usr/local/bin:/usr/bin:/bin',
            'PIP_NO_INDEX': '1', 'HF_HUB_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONPATH': str(deployed['workspace']), 'RAX_MCP_SOCKET': str(socket),
            'NODE_OPTIONS': '--import='+str(SUPPORT/'isolated_cli/transport.mjs'),
            'RAX_MODEL_SOCKET': str(Path(socket).parent/'model.sock')}
        if self.name == 'dsh':
            variables.update(DSH_HOME=str(home), RAX_DSH_INSTALL=str(self.native.installation(deployed['dsh'])),
                RAX_MODEL_KEY='rax-session-transport',
                RAX_MODEL_PROTOCOL='chat' if config.agent_provider == 'openrouter' else 'messages')
        else:
            variables.update(self.spec.environment(home, install=self.spec.installation(Path(value['binary']))))
        return variables

    def sandbox(self, config, deployed, command, *, name=None):
        value = json.loads(Path(deployed['login']).read_text())
        home = Path(deployed[self.HOME_KEY])
        install = (self.native.installation(deployed['dsh']) if self.name == 'dsh'
                   else self.spec.installation(Path(value['binary'])))
        limits = config.training
        published = list(lifecycle.published_mounts(config, deployed))
        workspace = Path(deployed['workspace'])
        # Both names refer to the workspace. Nested data mounts must cover both
        # views so relative recording/demo paths remain read-only and readable.
        for source, destination, readonly in list(published):
            relative = Path(destination).relative_to('/workspace')
            target = workspace/relative
            if not target.resolve().is_relative_to(workspace.resolve()):
                raise ValueError('Published mount target must stay inside the workspace')
            target.parent.mkdir(parents=True, exist_ok=True)
            if Path(source).is_dir():
                target.mkdir(exist_ok=True)
            else:
                target.touch(exist_ok=True)
            for parent in target.parents:
                if not parent.resolve().is_relative_to(workspace.resolve()):
                    break
                parent.chmod(parent.stat().st_mode | 0o010)
            published.append((source, str(target), readonly))
        args = ['docker', 'run', '--rm', '--pull=never', '--name', name or 'rax-agent-'+uuid.uuid4().hex,
            '--network=none', '--read-only', '--user', f'{os.getuid()}:{os.getgid()}',
            '--cap-drop=ALL', '--security-opt=no-new-privileges', '--ipc=private', '--init',
            '--pids-limit', str(limits.pids), '--memory', f'{limits.memory_mb}m',
            '--memory-swap', f'{limits.memory_mb}m', '--cpus', str(limits.cpus), '--log-driver=none',
            '--ulimit', 'core=0', '--tmpfs', f'/tmp:rw,nosuid,nodev,size={limits.scratch_mb}m,mode=1777',
            '--workdir', str(deployed['workspace']), *thread_env(limits.cpus)]
        mounts = [(deployed['workspace'], str(deployed['workspace']), False),
                  (deployed['workspace'], '/workspace', False), (home, str(home), False),
                  (install, str(install), True),
                  (Path(value['node']).parent, str(Path(value['node']).parent), True),
                  (config.root/'container-support', str(SUPPORT), True),
                  (config.root/'container-support/mcp_bridge.py', '/tmp/rax-bridge/mcp_bridge.py', True),
                  (agent_host.socket_path(config.root).parent, str(agent_host.socket_path(config.root).parent), True),
                  *published]
        # Preserve native absolute recording links as well as container workspace paths.
        mounts += [(config.root/'published', str(config.root/'published'), True)]
        mounts += [(source, str(Path(source).resolve()), True) for source, _ in reference_sources(config)]
        if self.spec and hasattr(self.spec, 'runtime_mounts'):
            mounts += [(source, str(destination), False) for source, destination in self.spec.runtime_mounts(home)]
        destinations = set()
        for source, destination, readonly in mounts:
            if destination in destinations:
                continue
            destinations.add(destination)
            if any(c in str(source)+str(destination) for c in (',', '\n')):
                raise ValueError('Invalid container mount path')
            args += ['--mount', f'type=bind,src={Path(source).resolve()},dst={destination}'+(',readonly' if readonly else '')]
        for key, item in self.agent_environment(config, deployed, agent_host.socket_path(config.root)).items():
            args += ['--env', key+'='+item]
        return [*args, '--entrypoint', command[0], config.image, *command[1:]]

    def agent_command(self, config, deployed, *, prompt, interactive=False, json_output=False, name=None):
        if interactive:
            raise ValueError(f'{self.name} uses its headless CLI')
        value = json.loads(Path(deployed['login']).read_text())
        command = (self.native.agent_command(config, deployed, prompt=prompt) if self.name == 'dsh'
                   else [value['node'], str(deployed['binary']),
                         *self.spec.arguments(prompt, deployed.get('resume_session'))])
        return self.sandbox(config, deployed, command, name=name)

    @contextmanager
    def relay(self, config, deployed):
        if self.name != 'dsh':
            with self.native.relay(config, deployed) as server:
                yield server
            return
        credential = DeepSeekCredential(self.native, deployed)
        value = json.loads(Path(deployed['login']).read_text())
        socket = agent_host.socket_path(config.root).parent/'model.sock'
        if config.agent_provider == 'openrouter':
            with serve(socket, credential=credential, model=value['model'], effort=config.agent_effort,
                       audit=Path(deployed[self.HOME_KEY])/'request-audit.jsonl') as server:
                yield server
        else:
            from harness.claude_cli.messages_relay import MessagesRelay
            server = MessagesRelay(socket, credential, base_url=credential.endpoint,
                models=[value['model']], client_write_timeout=CALL_SECONDS+300,
                allow_private_http=config.agent_provider == 'custom')
            server.start()
            try:
                yield server
            finally:
                server.close()

    def launch(self, config, deployed, *, gemini_key_file=None, prompt, interactive=False,
               capture=False, agent_runs=None, **unused):
        if interactive or capture:
            raise ValueError('Use non-interactive streaming mode')
        DockerSandbox(config.image, config.training).preflight()
        workspace_storage.validate(config.root, config.workspace_mb,
                                   deployed['workspace'], deployed[self.HOME_KEY])
        name = 'rax-agent-'+uuid.uuid4().hex
        socket = agent_host.socket_path(config.root)
        socket.unlink(missing_ok=True)
        self.write_home(config, deployed, socket)
        agent = server = None
        with lifecycle.open_frontend_log(config.root) as log, self.relay(config, deployed):
            try:
                args = [sys.executable, '-m', 'services.controller.frontend',
                    '--config', str(deployed['configuration']), '--workspace', str(deployed['workspace']),
                    '--socket', str(socket), '--agent-container', name]
                if gemini_key_file:
                    args += ['--key-file', str(gemini_key_file)]
                server = lifecycle.start_frontend(args, log)
                deadline = time.monotonic()+30
                while not socket.exists():
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise lifecycle.frontend_failure(server, log, 30)
                    time.sleep(.05)
                def run(text):
                    nonlocal agent
                    agent = subprocess.Popen(self.agent_command(config, deployed, prompt=text, name=name),
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
                    for line in agent.stdout:
                        self.native.capture_event(deployed[self.HOME_KEY], line)
                        print(line, end='', flush=True)
                    return agent.wait()
                code = agent_runs(run) if agent_runs else run(prompt)
                return subprocess.CompletedProcess(agent.args, code)
            finally:
                lifecycle.stop_services(name, agent, server, None)
