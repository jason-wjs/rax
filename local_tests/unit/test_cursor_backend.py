"""Cursor adapter contracts. No account calls or inference; native CLI smoke is separate."""
from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
import os
import shlex
import socket
from pathlib import Path
import subprocess
import sys

import pytest

from harness.cursor_cli import auto_research as cursor
from services.cursor_usage import UsageReader
from services.dashboard.sessions import ResearchSessions
from test_controller_backend import configuration


# Metadata from the captured official 2026.10.19 stream-json result. The CLI's
# turnEnded handler subtracts cache counts before emitting result.usage.inputTokens.
SESSION = '0f7a7050-9501-4d59-ac1b-cba83816e278'
RESULT = {'type': 'result', 'subtype': 'success', 'is_error': False,
          'session_id': SESSION, 'request_id': '9f68c689-79d2-4d45-a3d6-9eae7c087b7a',
          'duration_ms': 13285, 'duration_api_ms': 13285,
          'usage': {'inputTokens': 21745, 'outputTokens': 205,
                    'cacheReadTokens': 1152, 'cacheWriteTokens': 0}}


@pytest.fixture
def deployed(tmp_path, monkeypatch):
    # Cursor's ancestor scan accepts even a .git directory that Git rejects.
    (tmp_path/'.git').mkdir()
    config = replace(configuration(tmp_path), agent_cli='cursor',
                     agent_provider='cursor-account', agent_effort='xhigh')
    config.root.mkdir()
    auth = tmp_path/'operator-config/cursor'
    auth.mkdir(parents=True)
    (auth/'auth.json').write_text('fixture-private-account-token')
    (auth/'auth.json').chmod(0o600)
    binary = tmp_path/'bin/agent'
    binary.parent.mkdir()
    binary.touch()
    secret = tmp_path/'other-provider.key'
    secret.write_text('fixture-other-provider-token')
    def prepare(config, name):
        workspace, home = config.root/'agent-workspace', config.root/name
        workspace.mkdir()
        home.mkdir()
        (config.root/'published').mkdir()
        (config.root/'configuration.json').write_text('{}')
        return {'workspace': workspace, 'home': home,
                'configuration': config.root/'configuration.json'}
    monkeypatch.setattr(cursor, 'prepare_workspace', prepare)
    monkeypatch.setattr(cursor, 'reference_sources', lambda config: [])
    monkeypatch.setattr(cursor.agent_host, 'reexposed_paths', lambda: [])
    monkeypatch.setattr(cursor.agent_host, 'hidden_paths', lambda *args, **kwargs: [secret])
    monkeypatch.setattr(cursor.agent_host, 'socket_path', lambda root: tmp_path/'socket/mcp.sock')
    monkeypatch.setattr(cursor.subprocess, 'check_output', lambda *args, **kwargs: '2026.10.19-fixture\n')
    value = cursor.deploy(config, binary=binary, auth_home=auth)
    return config, value, auth, secret


def test_command_keeps_requested_model_and_resumes_recorded_session(deployed):
    config, value, _, _ = deployed
    command = cursor.agent_command(config, value, prompt='Start robot work')
    assert command[command.index('--model')+1] == cursor.MODEL
    assert command[command.index('--sandbox')+1] == 'enabled'
    assert command[command.index('--output-format')+1] == 'stream-json'
    assert command[-2:] == ['--', 'Start robot work']
    assert '--resume' not in command
    assert '--approve-mcps' not in command
    value['resume_session'] = SESSION
    command = cursor.agent_command(config, value, prompt='Continue')
    assert command[command.index('--resume')+1] == SESSION
    with pytest.raises(ValueError, match='non-interactive'):
        cursor.agent_command(config, value, prompt='', interactive=True)
    # This proves argument preservation, not vendor acceptance of the 500k selection.


def test_workspace_policy_protects_account_and_controls(deployed):
    config, value, auth, secret = deployed
    workspace, home = value['workspace'], value[cursor.HOME_KEY]
    policy = json.loads((workspace/'.cursor/cli.json').read_text())['permissions']
    sandbox = json.loads((workspace/'.cursor/sandbox.json').read_text())
    for path in (auth, home, secret):
        assert f'Read({path})' in policy['deny']
        assert f'Read({path}/**)' in policy['deny']
    for path in (auth, home, secret, workspace/'.cursor', workspace/'.git'):
        assert f'Write({path})' in policy['deny']
        assert f'Write({path}/**)' in policy['deny']
    assert str(workspace/'.cursor') in sandbox['additionalReadonlyPaths']
    assert str(workspace/'.git') in sandbox['additionalReadonlyPaths']
    assert str(config.root/'published') in sandbox['additionalReadPaths']
    assert sandbox['networkPolicy'] == {'default': 'deny', 'allow': [], 'deny': ['*']}
    hook = json.loads((workspace/'.cursor/hooks.json').read_text())['hooks']['subagentStart'][0]
    assert hook['failClosed'] is True
    mcp = json.loads((workspace/'.cursor/mcp.json').read_text())['mcpServers']['auto_research']
    assert mcp['args'][-1].endswith('/socket/mcp.sock')
    assert '-I' in mcp['args']


def test_account_is_referenced_without_copying_or_inherited_api_credentials(deployed, monkeypatch):
    config, value, auth, _ = deployed
    for name in ('CURSOR_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'DEEPSEEK_API_KEY', 'HTTP_PROXY'):
        monkeypatch.setenv(name, 'fixture-inherited-secret')
    env = cursor.agent_environment(config, value, Path('/tmp/fixture.sock'))
    assert env['XDG_CONFIG_HOME'] == str(auth.parent)
    assert env['CURSOR_CONFIG_DIR'] == str(value[cursor.HOME_KEY])
    assert env['AGENT_CLI_CREDENTIAL_STORE'] == 'file'
    assert all(name not in env for name in ('CURSOR_API_KEY', 'ANTHROPIC_AUTH_TOKEN',
                                           'DEEPSEEK_API_KEY', 'HTTP_PROXY'))
    assert not list(config.root.rglob('auth.json'))
    assert all('fixture-private-account-token' not in p.read_text()
               for p in config.root.rglob('*') if p.is_file())
    login = json.loads(value['login'].read_text())
    assert login['auth_home'] == str(auth)
    assert login['version'] == '2026.10.19-fixture'


def test_account_rejects_public_file_and_symlink(tmp_path):
    auth = tmp_path/'cursor'
    auth.mkdir()
    path = auth/'auth.json'
    path.write_text('fixture')
    path.chmod(0o644)
    with pytest.raises(ValueError, match='private'):
        cursor.account_home(auth)
    path.chmod(0o600)
    assert cursor.account_home(auth) == auth
    target = tmp_path/'private-auth'
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match='private'):
        cursor.account_home(auth)


def test_custom_endpoint_and_missing_cursor_binary_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='native account'):
        cursor.deploy(None, base_url='https://provider.invalid')
    with pytest.raises(RuntimeError, match='agent-binary'):
        cursor.cursor_binary(tmp_path/'absent')


def test_native_subagent_hook_denies(tmp_path):
    hook = Path(cursor.__file__).with_name('deny_subagents.py')
    result = subprocess.run([sys.executable, '-I', str(hook)], input='{}', text=True,
                            capture_output=True, check=True)
    assert json.loads(result.stdout)['permission'] == 'deny'


def test_export_includes_only_selected_conversation_store(tmp_path):
    selected = tmp_path/'chats/workspace-hash'/SESSION
    selected.mkdir(parents=True)
    for name in ('store.db', 'store.db-wal'):
        (selected/name).write_text('conversation')
    other = selected.parent/'another-session'
    other.mkdir()
    (other/'store.db').write_text('other conversation')
    (tmp_path/'auth.json').write_text('private token')
    (selected/'private-link').symlink_to(tmp_path/'auth.json')
    (tmp_path/'chats/symlink-hash').symlink_to(selected.parent, target_is_directory=True)
    paths = cursor.session_files(tmp_path, SESSION)
    assert set(paths) == {selected.relative_to(tmp_path)/'store.db',
                          selected.relative_to(tmp_path)/'store.db-wal'}
    assert cursor.session_files(tmp_path, 'unknown') == []


def test_recorded_init_result_and_errors_drive_resume_and_usage(tmp_path):
    for line in ('progress message', '[]', '{'):
        cursor.capture_event(tmp_path, line)
    assert cursor.latest_session(tmp_path) is None
    cursor.capture_event(tmp_path, json.dumps({'type': 'system', 'subtype': 'init',
        'session_id': SESSION, 'model': 'Grok 4.7  High Fast\u200b\u200b'}))
    assert cursor.latest_session(tmp_path) == SESSION
    assert cursor.last_error(tmp_path) is None
    cursor.capture_event(tmp_path, json.dumps(RESULT))
    cursor.capture_event(tmp_path, json.dumps(RESULT))
    measured = UsageReader().read(tmp_path)
    assert measured['completed_turns'] == 1
    assert measured['token_usage_complete'] is True
    assert measured['token_usage'] == {'input_tokens': 22897, 'cached_input_tokens': 1152,
        'cache_write_input_tokens': 0, 'output_tokens': 205, 'total_tokens': 23102}
    assert measured['duration_ms'] == measured['duration_api_ms'] == 13285
    assert measured['agent_steps'] is None and measured['model_calls'] is None
    cursor.capture_event(tmp_path, json.dumps({'type': 'error', 'message': 'Model unavailable'}))
    assert cursor.last_error(tmp_path) == 'Model unavailable'
    cursor.capture_event(tmp_path, json.dumps({'type': 'result', 'is_error': True,
        'subtype': 'error_during_execution', 'result': 'Request rejected'}))
    assert cursor.last_error(tmp_path) == 'Request rejected'


def test_usage_deduplicates_per_session_and_marks_missing_usage_unknown(tmp_path):
    assert UsageReader().read(tmp_path)['token_usage'] is None
    cursor.capture_event(tmp_path, json.dumps(RESULT))
    cursor.capture_event(tmp_path, json.dumps({**RESULT, 'session_id': 'second-session'}))
    cursor.capture_event(tmp_path, json.dumps({**RESULT, 'request_id': 'unmeasured', 'usage': None}))
    measured = UsageReader().read(tmp_path)
    assert measured['completed_turns'] == 3
    assert measured['token_usage_complete'] is False
    assert measured['token_usage']['total_tokens'] == 46204


def test_resume_requires_conversation_id_and_reuses_recorded_id(deployed, monkeypatch):
    config, value, _, _ = deployed
    monkeypatch.setattr(cursor, 'native_session', lambda *args: None)
    monkeypatch.setattr(cursor, 'expose_published', lambda *args: None)
    monkeypatch.setattr(cursor, 'refresh_reviews', lambda *args: None)
    with pytest.raises(RuntimeError, match='No Cursor conversation'):
        cursor.redeploy(config)
    cursor.capture_event(value[cursor.HOME_KEY], json.dumps({'type': 'system', 'subtype': 'init',
                                                          'session_id': SESSION}))
    resumed = cursor.redeploy(config)
    assert resumed['resume_session'] == SESSION
    assert resumed['cursor'] == value['cursor']


def test_dashboard_selects_cursor_usage_reader(tmp_path, monkeypatch):
    session = tmp_path/'session'
    home = session/'cursor-home'
    home.mkdir(parents=True)
    cursor.capture_event(home, json.dumps(RESULT))
    dashboard = ResearchSessions(tmp_path, lambda: [])
    monkeypatch.setattr(dashboard, 'directory', lambda *args, **kwargs: session)
    monkeypatch.setattr(dashboard, 'summary', lambda *args: {})
    monkeypatch.setattr(dashboard, 'command', lambda *args: [])
    monkeypatch.setattr(dashboard, 'live_observation', lambda *args: None)
    assert dashboard.detail('fixture')['usage']['token_usage']['total_tokens'] == 23102


def test_rewriting_home_preserves_only_official_identity_metadata(deployed):
    config, value, _, _ = deployed
    home = value[cursor.HOME_KEY]
    settings = home/'cli-config.json'
    identity = {'authId': 'fixture-account', 'userId': 123, 'email': 'fixture@example.invalid',
                'teamId': 7, 'activeTeamId': 7, 'organizationId': 'org-fixture',
                'displayName': 'Fixture', 'teamName': 'Fixture team'}
    settings.write_text(json.dumps({'authInfo': {**identity, 'accessToken': 'must-not-copy'},
        'approvalMode': 'unrestricted', 'sandbox': {'readBoundary': 'system'},
        'serverConfigCache': {'cliSandboxDefaultEnabled': False}}))
    cache = home/'statsig-cache.json'
    cache.write_text('fixture-vendor-cache-unchanged')
    cursor.write_home(config, value)
    written = json.loads(settings.read_text())
    assert written['authInfo'] == identity
    assert written['approvalMode'] == 'allowlist'
    assert written['sandbox']['readBoundary'] == 'workspace'
    assert 'serverConfigCache' not in written
    assert cache.read_text() == 'fixture-vendor-cache-unchanged'
    cursor.write_home(config, value)
    assert json.loads(settings.read_text())['authInfo'] == identity


def test_preflight_survives_vendor_shell_join_and_requires_actual_output(deployed, monkeypatch):
    config, value, _, _ = deployed
    captured = []
    def sandbox_run(command, **kwargs):
        # The official sandbox-debug implementation joins these argv elements.
        assert command[1:4] == ['sandbox', 'run', '--']
        parsed = shlex.split(' '.join(command[4:]))
        assert parsed[:3] == ['/usr/bin/python3', '-I', '-c']
        assert len(parsed) == 4
        compile(parsed[3], '<cursor-boundary>', 'exec')
        assert str(value[cursor.HOME_KEY]/'boundary-canary.txt') in parsed[3]
        captured.append(command)
        return subprocess.CompletedProcess(command, 0, 'STDOUT:\nRAX_CURSOR_BOUNDARY_OK\n', '')
    monkeypatch.setattr(cursor.subprocess, 'run', sandbox_run)
    cursor.preflight(config, value)
    assert len(captured) == 1
    assert json.loads((value[cursor.HOME_KEY]/'boundary-check.json').read_text())['passed']
    assert not (value[cursor.HOME_KEY]/'boundary-canary.txt').exists()


@pytest.mark.parametrize('returncode,stdout', [
    (0, "Executing command: python3 -c print('RAX_CURSOR_BOUNDARY_OK')\n"),
    (0, 'prefixed RAX_CURSOR_BOUNDARY_OK\n'),
    (1, 'RAX_CURSOR_BOUNDARY_OK\n'),
])
def test_failed_boundary_stops_launch_before_frontend_or_model(deployed, monkeypatch, returncode, stdout):
    config, value, _, _ = deployed
    calls = []
    def sandbox_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout, '')
    def unexpected_start(*args, **kwargs):
        pytest.fail('A failed boundary must not start the frontend or model')
    monkeypatch.setattr(cursor.subprocess, 'run', sandbox_run)
    monkeypatch.setattr(cursor, 'bootstrap_metadata', lambda *args: None)
    monkeypatch.setattr(cursor, 'approve_mcp', lambda *args: None)
    monkeypatch.setattr(cursor, 'start_frontend', unexpected_start)
    monkeypatch.setattr(cursor.agent_host, 'start', unexpected_start)
    with pytest.raises(RuntimeError, match='no model run was started'):
        cursor.launch(config, value, prompt='Must not reach a model')
    assert len(calls) == 1
    assert calls[0][1:3] == ['sandbox', 'run']
    assert not json.loads((value[cursor.HOME_KEY]/'boundary-check.json').read_text())['passed']
    assert not (value[cursor.HOME_KEY]/'boundary-canary.txt').exists()


@pytest.mark.parametrize('mode,expected', [('success', None), ('reject', 'authentication failed'),
                                         ('exit', 'exited before'), ('stall', 'timed out')])
def test_official_bootstrap_protocol_has_no_session_or_generation(deployed, mode, expected):
    config, value, _, _ = deployed
    calls = value['workspace']/'metadata-methods.jsonl'
    binary = value['cursor']
    binary.write_text(f'''#!{sys.executable}
import json,sys,time
from pathlib import Path
for line in sys.stdin:
    request=json.loads(line)
    with Path({str(calls)!r}).open('a') as out:
        out.write(json.dumps(request['method'])+'\\n')
    if request['method']=='authenticate':
        if {mode!r}=='exit': sys.exit(2)
        if {mode!r}=='stall': time.sleep(10)
    response={{'jsonrpc':'2.0','id':request['id'],'result':{{}}}}
    if request['method']=='authenticate' and {mode!r}=='reject':
        response={{'jsonrpc':'2.0','id':request['id'],'error':{{'code':-32000}}}}
    print(json.dumps(response),flush=True)
''')
    binary.chmod(0o700)
    if expected:
        with pytest.raises(RuntimeError, match=expected):
            cursor.bootstrap_metadata(config, value, timeout=1)
    else:
        cursor.bootstrap_metadata(config, value, timeout=5)
    assert [json.loads(line) for line in calls.read_text().splitlines()] == ['initialize', 'authenticate']
    outcome = json.loads((value[cursor.HOME_KEY]/'bootstrap-check.json').read_text())
    assert outcome['passed'] is (mode == 'success')
    assert outcome['generation_requests'] == 0
    assert not cursor.latest_session(value[cursor.HOME_KEY])


def test_failed_bootstrap_stops_before_boundary_frontend_or_model(deployed, monkeypatch):
    config, value, _, _ = deployed
    def fail(*args):
        raise RuntimeError('metadata unavailable')
    def unexpected(*args, **kwargs):
        pytest.fail('Bootstrap failure must stop launch')
    monkeypatch.setattr(cursor, 'bootstrap_metadata', fail)
    monkeypatch.setattr(cursor, 'preflight', unexpected)
    monkeypatch.setattr(cursor, 'start_frontend', unexpected)
    monkeypatch.setattr(cursor.agent_host, 'start', unexpected)
    with pytest.raises(RuntimeError, match='metadata unavailable'):
        cursor.launch(config, value, prompt='Must not generate')


def test_boundary_accepts_visible_metadata_only_when_contents_are_denied(deployed, monkeypatch):
    config, value, _, _ = deployed
    (value['workspace']/'output').mkdir()
    protected = {value[cursor.HOME_KEY]/'boundary-canary.txt', Path(f'/proc/{os.getpid()}/cmdline')}
    read_bytes, glob = Path.read_bytes, Path.glob
    def denied(path):
        if path in protected:
            assert path.exists()  # Real Linux metadata remains visible under the vendor sandbox.
            raise PermissionError('fixture kernel read denial')
        return read_bytes(path)
    def no_network(*args, **kwargs):
        raise OSError('fixture network namespace')
    def sandbox_run(command, **kwargs):
        script = shlex.split(' '.join(command[4:]))[3]
        output = io.StringIO()
        with redirect_stdout(output):
            exec(compile(script, '<native-boundary-fixture>', 'exec'), {})
        return subprocess.CompletedProcess(command, 0, output.getvalue(), '')
    monkeypatch.setattr(Path, 'read_bytes', denied)
    monkeypatch.setattr(Path, 'glob', lambda path, pattern: [] if path == Path('/dev') else glob(path, pattern))
    monkeypatch.setattr(socket, 'create_connection', no_network)
    monkeypatch.setattr(cursor.subprocess, 'run', sandbox_run)
    cursor.preflight(config, value)
    assert json.loads((value[cursor.HOME_KEY]/'boundary-check.json').read_text())['passed']


@pytest.mark.parametrize('changed', [None, 'label', 'context', 'effort', 'fast', 'max_mode'])
def test_actual_init_rejects_silent_model_downgrade(deployed, changed):
    _, value, _, _ = deployed
    home = value[cursor.HOME_KEY]
    params = {'context': '500k', 'reasoning_effort': 'xhigh', 'fast': 'true'}
    label = 'Grok 4.7 500K Extra High Fast'
    if changed == 'label': label = 'Grok 4.7 High Fast'
    if changed == 'context': params['context'] = '256k'
    if changed == 'effort': params['reasoning_effort'] = 'high'
    if changed == 'fast': params['fast'] = 'false'
    settings = json.loads((home/'cli-config.json').read_text())
    settings.update(model={'maxMode': True}, maxMode=changed != 'max_mode',
        selectedModel={'modelId': 'grok-4.7', 'parameters': [{'id': k, 'value': v} for k, v in params.items()]})
    (home/'cli-config.json').write_text(json.dumps(settings))
    event = json.dumps({'type': 'system', 'subtype': 'init', 'session_id': SESSION, 'model': label})
    if changed:
        with pytest.raises(RuntimeError, match='did not initialize'):
            cursor.capture_event(home, event, requested_model=cursor.MODEL)
        assert cursor.last_error(home)
    else:
        cursor.capture_event(home, event, requested_model=cursor.MODEL)
    assert json.loads((home/'model-check.json').read_text())['passed'] is (changed is None)


def test_deploy_anchors_project_inside_rax_workspace(deployed):
    _, value, _, _ = deployed
    workspace = value['workspace']
    # check_output is stubbed by the fixture for Cursor's --version only.
    result = subprocess.run(['git', '-C', str(workspace), 'rev-parse', '--show-toplevel'],
                            capture_output=True, text=True, check=True)
    assert Path(result.stdout.strip()) == workspace
    assert (workspace/'.git/config').is_file()


@pytest.mark.parametrize('returncode', [0, 1])
def test_mcp_approval_is_local_and_limited_to_rax(deployed, monkeypatch, returncode):
    config, value, auth, _ = deployed
    calls = []
    def approve(command, **kwargs):
        calls.append(command)
        assert command == [str(value['cursor']), 'mcp', 'enable', 'auto_research']
        assert kwargs['cwd'] == value['workspace']
        assert kwargs['env']['CURSOR_CONFIG_DIR'] == str(value[cursor.HOME_KEY])
        assert kwargs['env']['XDG_CONFIG_HOME'] == str(auth.parent)
        return subprocess.CompletedProcess(command, returncode, '', '')
    monkeypatch.setattr(cursor.subprocess, 'run', approve)
    if returncode:
        with pytest.raises(RuntimeError, match='no model run was started'):
            cursor.approve_mcp(config, value)
    else:
        cursor.approve_mcp(config, value)
    assert len(calls) == 1
    result = json.loads((value[cursor.HOME_KEY]/'mcp-check.json').read_text())
    assert result['passed'] is (returncode == 0)
    assert result['approved_servers'] == ([] if returncode else ['auto_research'])
