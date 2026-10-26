"""Native DSH credentials, isolation, resume bookkeeping, and exact usage semantics."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from harness.backends import get
from harness.deepseek_cli import auto_research as dsh
from harness.deepseek_cli.login import Credential
from harness.deepseek_cli.runner import arguments
from services.dsh_usage import UsageReader
from test_controller_backend import configuration


def private_key(tmp_path):
    path = tmp_path/'model.key'
    path.write_text('validation-only-key')
    path.chmod(0o600)
    return path


def test_native_dsh_is_a_distinct_backend_and_refuses_live_formal(tmp_path):
    config = replace(configuration(tmp_path), agent_cli='dsh', agent_effort='max')
    assert get(config.agent_cli) is dsh
    for updates in ({'agent_formal': True}, {'subagents': 1}, {'exploration_envs': 2}):
        with pytest.raises(ValueError, match='DSH currently supports code formal'):
            replace(config, **updates)
    with pytest.raises(ValueError, match='agent_provider'):
        replace(config, agent_provider='claude-login')


def test_credentials_are_private_and_never_in_provider_configuration(tmp_path):
    key = private_key(tmp_path)
    credential = Credential('openrouter', key_file=key)
    route, rows = credential.patches('deepseek/deepseek-v4.1-flash-1017', 'max')
    value = rows[0]['config']['providers'][route]
    assert value['reasoning'] == 'max'
    assert value['models'][0]['reasoningEfforts']['max'] == 'max'
    assert value['models'][0]['input'] == ['text','image']
    assert value['api'] == 'openai-completions'
    assert key.read_text() not in json.dumps(rows)
    assert key.read_text() not in json.dumps(credential.record())
    key.chmod(0o644)
    with pytest.raises(ValueError, match='private'):
        Credential('openrouter', key_file=key)


def test_custom_credentials_never_allow_embedded_url_auth(tmp_path):
    key = private_key(tmp_path)
    with pytest.raises(ValueError, match='without embedded credentials'):
        Credential('custom', key_file=key, base_url='https://user:secret@host.invalid')
    credential = Credential('custom', key_file=key, base_url='http://127.0.0.1:8000')
    route, rows = credential.patches('deepseek-flash', 'max')
    assert route == 'rax-private'
    assert rows[0]['insert'][0]['config'] == {'baseURL':'http://127.0.0.1:8000',
        'thinking':'enabled', 'reasoningEffort':'max',
        'models': [{'id': 'deepseek-flash', 'name': 'deepseek-flash', 'inputModalities': ['text', 'image']}]}


def test_account_mode_uses_native_store_and_max_effort_without_api_key(tmp_path):
    (tmp_path/'.credentials.yaml').write_text('fixture-native-store')
    (tmp_path/'.credentials.yaml').chmod(0o600)
    credential = Credential('deepseek-account', auth_home=tmp_path, key_file=tmp_path/'unused-default-api.key')
    route, rows = credential.patches('deepseek-flash', 'max')
    assert route == 'deepseek-account' and credential.environment() == {}
    assert credential.record()['key_file'] is None
    assert rows[0]['config']['reasoningEffort'] == 'max'
    assert {r['id'] for r in rows if r.get('disabled') is False} == {'credentials','authorization','deepseek-account'}
    assert rows[-1]['config']['path'] == str(tmp_path/'.credentials.yaml')
    assert 'fixture-native-store' not in json.dumps(rows)


def test_shell_profile_keeps_network_and_gpu_isolated(tmp_path):
    boundary = {'bwrap':'/usr/bin/bwrap','workspace':str(tmp_path),'socket_directory':str(tmp_path/'socket'),
                'hidden':[],'reexposed':[]}
    result = arguments(boundary, ['/bin/bash','-c','echo OK'])
    assert '--unshare-net' in result and '--unshare-pid' in result
    assert result[result.index('--dev')+1] == '/dev'
    assert result[-3:] == ['/bin/bash','-c','echo OK']
    with pytest.raises(ValueError, match='Missing sandbox command'):
        arguments(boundary, [])


def test_resume_identity_and_usage_keep_native_step_and_cache_semantics(tmp_path):
    dsh.capture_event(tmp_path, json.dumps({'type':'session','sessionId':'session-unique'}))
    native = {'type':'status','phase':'step_end','turn':1,'step':1,
              'usage':{'inputTokens':6454,'outputTokens':256,'totalTokens':6966,'cacheReadTokens':256}}
    dsh.capture_event(tmp_path, json.dumps(native))
    dsh.capture_event(tmp_path, json.dumps(native))  # Replayed projection must not double-charge.
    assert dsh.latest_session(tmp_path) == 'session-unique'
    usage = UsageReader().read(tmp_path)
    assert usage['token_usage']['input_tokens'] == 6710
    assert usage['token_usage']['cached_input_tokens'] == 256
    assert usage['token_usage']['total_tokens'] == 6966
    assert usage['agent_steps'] == 1 and usage['token_usage_complete']
    native.update(step=2)
    native.pop('usage')
    dsh.capture_event(tmp_path, json.dumps(native))
    assert UsageReader().read(tmp_path)['token_usage_complete'] is False


def test_incomplete_usage_record_and_non_events_are_ignored(tmp_path):
    (tmp_path/'usage.jsonl').write_text('{"session_id":')
    dsh.capture_event(tmp_path, '[]')
    dsh.capture_event(tmp_path, 'not-json')
    assert UsageReader().read(tmp_path)['token_usage'] is None


def test_frozen_dsh_package_carries_conversation_without_credentials(tmp_path):
    from services.transfer import pack
    home, workspace = tmp_path/'home', tmp_path/'workspace'
    workspace.mkdir()
    session = home/'sessions/workspace/native-id/session.v3.jsonl'
    session.parent.mkdir(parents=True)
    session.write_text('{"fixture":"native persisted conversation"}\n')
    (home/'.credentials.yaml').write_text('private-account-fixture')
    (home/'profiles').mkdir()
    (home/'profiles/operator.patch.yml').write_text('private-operator-fixture')
    package = tmp_path/'package'
    _, record = pack(package, workspace=workspace, parts=(), home=home,
        session_files=dsh.session_files(home, 'native-id'), home_files=dsh.HOME_FILES)
    assert record['session_files'] == ['sessions/workspace/native-id/session.v3.jsonl']
    assert record['home_files'] == []
    assert (package/'session'/record['session_files'][0]).read_text() == session.read_text()
    assert not (package/'session/.credentials.yaml').exists()


@pytest.mark.parametrize('model', ['deepseek-flash', 'deepseek-v4.1-flash'])
def test_custom_model_declares_image_input(tmp_path, model):
    key = private_key(tmp_path)
    credential = Credential('custom', key_file=key, base_url='https://relay.invalid')
    route, rows = credential.patches(model, 'max')
    configured = rows[0]['insert'][0]['config']['models']
    selected = next(entry for entry in configured if entry['id'] == model)
    assert selected['inputModalities'] == ['text', 'image']
    assert key.read_text() not in json.dumps(rows)
