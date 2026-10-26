"""The pilot's three agents share task configuration and operator entry points."""
from dataclasses import replace
from pathlib import Path

import pytest

from harness import backends, cli_runtime
from scripts.configure_auto_research import arguments
from services.controller.config import Configuration
from services.dashboard.sessions import ResearchSessions
from services.dexjoco.tasks import OFFICIAL_TASKS
from test_controller_backend import configuration
from test_session_resume import launcher_main, stopped_session
from harness.codex_cli import auto_research as launcher
from harness.claude_cli import auto_research as claude


@pytest.mark.parametrize('task', list(OFFICIAL_TASKS))
@pytest.mark.parametrize('cli,provider,effort', [
    ('dsh', 'custom', 'max'), ('claude', 'custom', 'max'), ('cursor', 'cursor-account', 'xhigh')])
def test_all_eleven_tasks_accept_the_three_pilot_agents(tmp_path, task, cli, provider, effort):
    config = replace(configuration(tmp_path), task=task, benchmark='dexjoco', agent_cli=cli,
                     agent_provider=provider, agent_effort=effort, exploration_envs=1, subagents=0,
                     exploration_seeds=tuple(range(16)), formal_episodes=3, formal_workers=1,
                     demonstration_context='terminal_state', external_perception=False)
    assert config.agent_cli == cli and config.agent_provider == provider
    assert config.task == task and config.agent_effort == effort
    assert not config.agent_formal and config.exploration_envs == 1


def test_cursor_config_and_dashboard_reject_unsupported_modes(tmp_path):
    config = replace(configuration(tmp_path), agent_cli='cursor', agent_provider='cursor-account')
    for changes in ({'agent_formal': True}, {'exploration_envs': 2}, {'subagents': 1}):
        with pytest.raises(ValueError, match='supports code formal'):
            replace(config, **changes)
    with pytest.raises(ValueError, match='agent_provider'):
        replace(config, agent_provider='custom')
    dashboard = ResearchSessions(tmp_path, lambda: list(OFFICIAL_TASKS))
    settings = {'task': 'click_mouse', 'agent_cli': 'cursor', 'agent_provider': 'cursor-account'}
    assert dashboard.settings(settings)['agent_provider'] == 'cursor-account'
    assert dashboard.status()['agent_providers']['cursor'] == ['cursor-account']
    for changes in ({'agent_formal': 'on'}, {'exploration_envs': 2}, {'subagents': 1}):
        with pytest.raises(ValueError, match='supports code formal'):
            dashboard.settings({**settings, **changes})


def test_cursor_is_native_only_and_configure_accepts_account_provider(monkeypatch):
    args = arguments().parse_args(['--task', 'click_mouse', '--sim-gpu', '0',
        '--agent-cli', 'cursor', '--agent-provider', 'cursor-account'])
    assert args.agent_cli == 'cursor' and args.agent_provider == 'cursor-account'
    monkeypatch.setattr(cli_runtime, 'CONTAINER', True)
    with pytest.raises(ValueError, match='native execution only'):
        backends.get('cursor')


@pytest.mark.parametrize('legacy', [False, True])
def test_generic_claude_launch_options_reach_existing_adapter(tmp_path, monkeypatch, legacy):
    prefix = '--claude-' if legacy else '--agent-'
    config = launcher_main(tmp_path, monkeypatch, '--prepare-only', '--agent-binary', '/fixture/claude',
        prefix+'model', 'deepseek-v4.1-flash', prefix+'key-file', '/fixture/private.key',
        prefix+'base-url', 'https://relay.invalid')
    config.agent_cli, config.agent_provider = 'claude', 'custom'
    seen = {}
    def deploy(config, **kwargs):
        seen.update(kwargs)
        return {'workspace': config.root/'agent-workspace'}
    monkeypatch.setattr(claude, 'deploy', deploy)
    launcher.main()
    assert seen == {'provider': 'custom', 'model': 'deepseek-v4.1-flash',
                    'key_file': Path('/fixture/private.key'), 'claude': Path('/fixture/claude'),
                    'base_url': 'https://relay.invalid'}


def test_conflicting_generic_and_legacy_claude_options_fail_before_deploy(tmp_path, monkeypatch, capsys):
    launcher_main(tmp_path, monkeypatch, '--prepare-only', '--agent-model', 'model-a', '--claude-model', 'model-b')
    with pytest.raises(SystemExit) as result:
        launcher.main()
    assert result.value.code == 2
    assert '--agent-model conflicts with --claude-model' in capsys.readouterr().err


def test_resume_passes_the_selected_claude_binary(tmp_path, monkeypatch):
    config = launcher_main(tmp_path, monkeypatch, '--resume', '--agent-binary', '/fixture/claude')
    config.agent_cli, config.agent_provider = 'claude', 'custom'
    stopped_session(config.root)
    (config.root/'configuration.json').write_text('{}')
    seen = {}
    class AtDeployment(Exception):
        pass
    def redeploy(config, **kwargs):
        seen.update(kwargs)
        raise AtDeployment()
    monkeypatch.setattr(claude, 'redeploy', redeploy)
    with pytest.raises(AtDeployment):
        launcher.main()
    assert seen == {'claude': Path('/fixture/claude')}


@pytest.mark.parametrize('cli,provider', [('dsh', 'custom'), ('cursor', 'cursor-account'), ('kimi', 'custom')])
def test_high_effort_is_rejected_when_the_adapter_cannot_apply_it(tmp_path, cli, provider):
    with pytest.raises(ValueError, match='high effort is currently supported only by Codex and Claude'):
        replace(configuration(tmp_path), agent_cli=cli, agent_provider=provider, agent_effort='high')
    dashboard = ResearchSessions(tmp_path, lambda: list(OFFICIAL_TASKS))
    with pytest.raises(ValueError, match='high effort is currently supported only by Codex and Claude'):
        dashboard.settings({'task': 'hammer_nail', 'agent_cli': cli, 'agent_provider': provider,
                            'agent_effort': 'high'})
