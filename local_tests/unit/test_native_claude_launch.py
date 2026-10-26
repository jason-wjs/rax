"""Native Claude Code: started in its workspace with its own login, settings and robot tools,
and nothing it starts outlives it."""
import json
import os
from pathlib import PurePosixPath
import subprocess
import sys
import time

import pytest

from test_controller_backend import configuration
from harness import agent_host
from harness.claude_cli import auto_research as claude, experiment_login

# Records how it was started, leaves a daemonized child behind and exits.
FAKE_CLAUDE = '''#!/bin/sh
printf '%s\\n' "$@" > "$CLAUDE_CONFIG_DIR/argv"
env > "$CLAUDE_CONFIG_DIR/env"
pwd -P > "$CLAUDE_CONFIG_DIR/cwd"
( setsid sleep {marker} > /dev/null 2>&1 & )
exit 3
'''


@pytest.fixture
def session(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    config.root.mkdir(parents=True)
    workspace, home = tmp_path/'agent-workspace', tmp_path/'claude-home'
    workspace.mkdir()
    home.mkdir()
    marker = f'31.{os.getpid() % 1000:03d}'
    binary = tmp_path/'bin/claude'
    binary.parent.mkdir()
    binary.write_text(FAKE_CLAUDE.format(marker=marker))
    binary.chmod(0o755)
    token = tmp_path/'token'
    token.write_text('sk-ant-oat01-test\n')
    token.chmod(0o600)
    login = config.root/'agent-login.json'
    login.write_text(json.dumps({'agent_cli': 'claude', 'provider': 'claude-login', 'model': 'claude-opus-5-5',
                                 'key_file': str(token), 'account_name': None}))
    deployed = {'workspace': workspace, 'claude_home': home, 'claude': binary, 'login': login, 'subagents': 0}
    servers = []
    def frontend(config, deployed, socket, log, *, key_file=None):
        servers.append(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']))
        return servers[-1]
    monkeypatch.setattr(claude, 'start_frontend', frontend)
    probes = []
    def usage(path):  # The account still has usage left; no network.
        probes.append(path)
        return {'http_status': 200, 'status': 'allowed', 'five_hour': 0.1, 'seven_day': 0.2, 'reset': None}
    monkeypatch.setattr(experiment_login, 'usage', usage)
    monkeypatch.setenv('RAX_OPERATOR_ONLY', 'x')  # Nothing of the launcher's environment is inherited.
    yield config, deployed, marker, servers, probes
    for server in servers:
        if server.poll() is None:
            server.kill()
            server.wait()


def test_claude_runs_natively_and_leaves_nothing_running(session):
    config, deployed, marker, servers, probes = session
    home = deployed['claude_home'].resolve()
    result = claude.launch(config, deployed, prompt='go')
    assert result.returncode == 3  # The CLI's own exit status, passed through.
    argv = (home/'argv').read_text().splitlines()
    assert argv[0] == '-p' and argv[argv.index('--model') + 1] == 'claude-opus-5-5'
    assert argv[argv.index('--mcp-config') + 1] == str(home/'mcp.json') and '--strict-mcp-config' in argv
    denied = argv[argv.index('--disallowedTools') + 1:argv.index('--output-format')]
    assert {'WebFetch', 'WebSearch', 'Agent', 'Task', 'Workflow', 'ScheduleWakeup'} <= set(denied)  # No internet; subagent cap 0.
    assert argv[-2:] == ['--', 'go'] and '--resume' not in argv
    environment = dict(line.split('=', 1) for line in (home/'env').read_text().splitlines() if '=' in line)
    assert environment['CLAUDE_CONFIG_DIR'] == str(home) and environment['HOME'] == str(home)
    assert environment['CLAUDE_CODE_OAUTH_TOKEN'] == 'sk-ant-oat01-test'
    assert environment[agent_host.SOCKET_ENV] == str(agent_host.socket_path(config.root))
    assert 'RAX_OPERATOR_ONLY' not in environment and 'ANTHROPIC_API_KEY' not in environment
    # Cost settings (README, Claude Code): compaction at 250k, the one-hour cache for the main
    # agent and its subagents, and background subagents kept past the main turn.
    assert environment['CLAUDE_CODE_AUTO_COMPACT_WINDOW'] == '250000'
    assert environment['ENABLE_PROMPT_CACHING_1H'] == '1'
    assert environment['CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL'] == '1h'
    assert environment['CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS'] == '0'
    assert (home/'cwd').read_text().strip() == str(deployed['workspace'].resolve())
    # Claude Code's own sandbox confines the agent's commands; its settings are the session's.
    settings = json.loads((home/'settings.json').read_text())
    assert settings['sandbox']['enabled'] and settings['sandbox']['failIfUnavailable']
    assert settings['sandbox']['network']['allowedDomains'] == []
    assert not settings['sandbox']['allowUnsandboxedCommands']
    mcp = json.loads((home/'mcp.json').read_text())['mcpServers'][claude.MCP_SERVER]
    assert mcp['args'][-1] == str(agent_host.socket_path(config.root))
    # The daemonized child the CLI left is gone, the frontend is stopped, and no pid file remains.
    assert subprocess.run(['pgrep', '-f', f'sleep {marker}'], capture_output=True).returncode == 1
    assert servers[0].poll() is not None and not (config.root/claude.PID_FILE).exists()
    assert probes == [deployed['claude_home'].parent/'token']  # One usage probe after the run, with usage left.


def test_a_session_resumes_its_own_conversation(session):
    config, deployed, marker, servers, probes = session
    claude.launch(config, {**deployed, 'resume_session': 'thread-1'}, prompt='again')
    argv = (deployed['claude_home']/'argv').read_text().splitlines()
    assert argv[argv.index('--resume') + 1] == 'thread-1' and argv[-2:] == ['--', 'again']


def test_expert_demo_read_permissions_cover_symlink_and_target(session, tmp_path, monkeypatch):
    config, deployed, *_ = session
    target = tmp_path/'reference/experts'
    target.mkdir(parents=True)
    alias = deployed['workspace']/'runtime/experts'
    alias.parent.mkdir()
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(claude, 'reference_sources',
                        lambda _: [(target, PurePosixPath('runtime/experts'))])
    settings = claude.settings(config, deployed, tmp_path/'socket/mcp.sock')
    allowed = settings['permissions']['allow']
    assert f'Read(/{alias}/**)' in allowed
    assert f'Read(/{alias.resolve()}/**)' in allowed
    assert str(target) in settings['sandbox']['filesystem']['allowRead']
    assert str(target) not in settings['sandbox']['filesystem']['allowWrite']
    assert settings['permissions']['defaultMode'] == 'acceptEdits'


def test_an_exhausted_usage_window_is_waited_out_in_the_same_conversation(session, monkeypatch):
    config, deployed, marker, servers, probes = session
    monkeypatch.setattr(claude, 'latest_claude_session', lambda home: 'conversation-1')
    reset = int(time.time()) + 120
    rows = [{'http_status': 429, 'status': 'rejected', 'reset': reset}, {'http_status': 200, 'status': 'allowed'}]
    runs, waits, slept = [], [], []
    def run(prompt):
        runs.append(prompt)
        return 1
    code = claude.until_usage_resets(run, deployed=deployed, prompt='go', interactive=False,
                                     on_wait=lambda until, waited: waits.append((until, waited)),
                                     sleep=slept.append, probe=lambda path: rows.pop(0))
    assert code == 1 and runs == ['go', claude.USAGE_PROMPT] and not rows
    assert 175 < slept[0] <= 180 and waits == [(reset, 0.0), (None, slept[0])]
    assert deployed['resume_session'] == 'conversation-1'
    always = lambda path: {'http_status': 429, 'status': 'rejected', 'reset': int(time.time())}
    runs.clear()
    claude.until_usage_resets(run, deployed=deployed, prompt='go', interactive=False, sleep=lambda s: None,
                              probe=always)
    assert len(runs) == 1 + claude.MAX_USAGE_WAITS


@pytest.mark.parametrize('row', [
    {'http_status': 200, 'status': 'allowed'},
    {'http_status': 429, 'status': 'allowed', 'reset': 1800000000},  # An ordinary rate limit.
    {'http_status': 401, 'status': None},  # A rejected token: the run's own error stands.
    {'http_status': None, 'status': 'unreachable', 'error': 'OSError: unreachable'},
])
def test_only_an_exhausted_usage_window_waits(session, row):
    config, deployed, marker, servers, probes = session
    runs = []
    code = claude.until_usage_resets(lambda prompt: runs.append(prompt) or 7, deployed=deployed, prompt='go',
                                     interactive=False, sleep=lambda s: pytest.fail('waited'),
                                     probe=lambda path: row)
    assert code == 7 and runs == ['go']


def test_interactive_runs_and_other_providers_are_not_probed(session):
    config, deployed, marker, servers, probes = session
    probe = lambda path: pytest.fail('probed')
    assert claude.until_usage_resets(lambda prompt: 5, deployed=deployed, prompt='go', interactive=True,
                                     probe=probe) == 5
    value = json.loads(deployed['login'].read_text())
    deployed['login'].write_text(json.dumps({**value, 'provider': 'openrouter'}))
    assert claude.limited_until(deployed, probe) is None
