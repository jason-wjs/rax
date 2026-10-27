"""Resuming an auto-research session: same workspace and state, the agent's own conversation."""
import json
import os
import time

import pytest

from harness.codex_cli import auto_research as launcher
from bench import BENCH, TASK


def rollout(path, thread, source='user', mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'type': 'session_meta', 'payload': {'id': thread, 'thread_source': source}}) + '\n')
    if mtime:
        os.utime(path, (mtime, mtime))


def test_newest_top_level_codex_thread_is_resumed(tmp_path):
    sessions = tmp_path/'sessions/2026/09/29'
    rollout(sessions/'rollout-a.jsonl', 'old-thread', mtime=100)
    rollout(sessions/'rollout-b.jsonl', 'main-thread', mtime=200)
    rollout(sessions/'rollout-c.jsonl', 'subagent-thread', source='subagent', mtime=300)
    assert launcher.latest_codex_thread(tmp_path) == 'main-thread'
    with pytest.raises(RuntimeError, match='No Codex conversation'):
        launcher.latest_codex_thread(tmp_path/'empty')


def test_newest_top_level_claude_session_is_resumed(tmp_path):
    from harness.claude_cli.auto_research import latest_claude_session
    projects = tmp_path/'projects/-workspace'
    (projects/'aaa/subagents').mkdir(parents=True)
    (projects/'aaa.jsonl').write_text('{}\n'); os.utime(projects/'aaa.jsonl', (100, 100))
    (projects/'bbb.jsonl').write_text('{}\n'); os.utime(projects/'bbb.jsonl', (200, 200))
    (projects/'aaa/subagents/zzz.jsonl').write_text('{}\n')
    assert latest_claude_session(tmp_path) == 'bbb'


def test_lifecycle_attempts(tmp_path):
    record = tmp_path/'agent-lifecycle.json'
    with pytest.raises(ValueError, match='never launched'):
        launcher.resumable_lifecycle(record)
    record.write_text(json.dumps({'status': 'prepared-only', 'finished_at': 1}))
    with pytest.raises(ValueError, match='only prepared'):
        launcher.resumable_lifecycle(record)
    record.write_text(json.dumps({'pid': 1, 'started_at': 0, 'finished_at': 100, 'status': 'stopped',
                                  'attempts': [{'started_at': 0, 'finished_at': 50, 'status': 'exited'}]}))
    attempts = launcher.resumable_lifecycle(record)
    assert [a['status'] for a in attempts] == ['exited', 'stopped'] and 'attempts' not in attempts[-1]
    assert [(a['started_at'], a['finished_at']) for a in attempts] == [(0, 50), (0, 100)]


def test_a_crashed_launch_ends_at_its_last_activity(tmp_path):
    record = tmp_path/'agent-lifecycle.json'
    record.write_text(json.dumps({'pid': 1, 'started_at': 1000, 'status': 'running'}))
    (tmp_path/'frontend.log').write_text('...')
    os.utime(tmp_path/'frontend.log', (1600, 1600))
    attempts = launcher.resumable_lifecycle(record)
    assert attempts[-1]['status'] == 'crashed' and attempts[-1]['finished_at'] == 1600


def test_claude_resumes_its_session_id(tmp_path):
    from types import SimpleNamespace
    from harness.claude_cli.auto_research import agent_command
    deployed = {'claude': 'claude', 'claude_home': tmp_path, 'model': 'claude-opus-5-5', 'resume_session': 'bbb'}
    args = agent_command(SimpleNamespace(), deployed, prompt='go')
    assert args[args.index('--resume')+1] == 'bbb' and args[-2:] == ['--', 'go']
    assert '--resume' not in agent_command(SimpleNamespace(), {**deployed, 'resume_session': None}, prompt='go')


def test_dashboard_offers_resume_only_for_stopped_auto_research_sessions(tmp_path, monkeypatch):
    from services.dashboard.sessions import ResearchSessions as Sessions
    model = Sessions(tmp_path/'project', lambda: [TASK])
    directory = tmp_path/'session'
    (directory/'agent-workspace').mkdir(parents=True)
    auto = {'mode': 'auto-research'}
    monkeypatch.setattr(Sessions, 'alive', staticmethod(lambda d, l: l.get('status') == 'running'))
    assert model.resumable(directory, {'status': 'stopped', 'finished_at': time.time()}, auto)
    assert not model.resumable(directory, {'status': 'running'}, auto)
    assert not model.resumable(directory, {}, auto)
    assert not model.resumable(directory, {'status': 'prepared-only'}, auto)
    assert not model.resumable(directory, {'status': 'exited', 'kind': 'formal_batch'}, auto)
    assert not model.resumable(tmp_path/'missing', {'status': 'exited'}, auto)


def relaunch(codes, *, interactive=False, open_after=None):
    calls, lifecycle, deployed = [], {}, {}
    def start(prompt):
        calls.append((prompt, deployed.get('resume_session')))
        return codes[len(calls)-1]
    still_open = lambda: open_after is None or len(calls) < open_after
    code = launcher.run_agent(start, deployed=deployed, prompt='go', interactive=interactive,
                              open_exploration=still_open, latest=lambda: f'conv{len(calls)}',
                              lifecycle=lifecycle, save=lambda: None)
    return code, calls, lifecycle


def test_an_early_exit_with_exploration_open_resumes_the_conversation():
    code, calls, lifecycle = relaunch([0, 0, 0], open_after=2)
    assert code == 0 and lifecycle['relaunches'] == 1
    assert calls == [('go', None), (launcher.CONTINUE_PROMPT, 'conv1')]


def test_relaunches_are_bounded():
    code, calls, lifecycle = relaunch([0]*10)
    assert len(calls) == launcher.MAX_RELAUNCHES+1 and lifecycle['relaunches'] == launcher.MAX_RELAUNCHES


def test_no_relaunch_when_interactive():
    assert len(relaunch([0, 0], interactive=True)[1]) == 1


def failing(codes, open_runs):
    calls, lifecycle, deployed, pauses = [], {}, {}, []
    def start(prompt):
        calls.append((prompt, deployed.get('resume_session')))
        return codes[len(calls)-1]
    code = launcher.run_agent(start, deployed=deployed, prompt='go', interactive=False,
                              open_exploration=lambda: len(calls) < open_runs, latest=lambda: f'conv{len(calls)}',
                              lifecycle=lifecycle, save=lambda: None,
                              last_error=lambda: 'ERROR: Selected model is at capacity. Please try a different model.',
                              sleep=pauses.append)
    return code, calls, lifecycle, pauses


def test_a_failed_agent_resumes_its_conversation_after_a_pause_with_the_error():
    error = 'ERROR: Selected model is at capacity. Please try a different model.'
    code, calls, lifecycle, pauses = failing([1, 0], open_runs=2)
    assert code == 0 and pauses == [launcher.ERROR_PAUSE_SECONDS]
    assert calls == [('go', None), (launcher.ERROR_PROMPT.format(error=error), 'conv1')]
    assert [e['error'] for e in lifecycle['agent_errors']] == [error]
    assert 'relaunches' not in lifecycle  # Error resumes do not use up the turn-end relaunches.


def test_failures_in_a_row_are_bounded():
    code, calls, lifecycle, pauses = failing([1]*20, open_runs=20)
    assert code == 1 and len(calls) == launcher.MAX_ERROR_RELAUNCHES+1
    assert len(lifecycle['agent_errors']) == len(calls) and len(pauses) == launcher.MAX_ERROR_RELAUNCHES


def test_a_finished_turn_resets_the_failures_in_a_row():
    """Five failures, a run that ends its turn normally, then five more: never six in a row."""
    codes = [1]*5 + [0] + [1]*5 + [0]
    code, calls, lifecycle, pauses = failing(codes, open_runs=len(codes))
    assert code == 0 and len(calls) == len(codes) and len(pauses) == 10 and lifecycle['relaunches'] == 1


def test_a_failed_agent_is_not_relaunched_once_exploration_is_closed():
    code, calls, lifecycle, pauses = failing([1, 0], open_runs=1)
    assert code == 1 and len(calls) == 1 and pauses == [] and 'agent_errors' not in lifecycle


@pytest.mark.parametrize('returncode', [0, 1])
@pytest.mark.parametrize('raises', [True, False])
def test_no_relaunch_without_a_conversation(returncode, raises):
    def missing():
        if raises:
            raise RuntimeError('none')
        return None
    calls, pauses = [], []
    code = launcher.run_agent(lambda p: calls.append(p) or returncode, deployed={}, prompt='go',
                              interactive=False, open_exploration=lambda: True, latest=missing,
                              lifecycle={}, save=lambda: None, sleep=pauses.append)
    assert code == returncode and calls == ['go'] and pauses == []


def test_resume_recovers_episodes_left_live_and_returns_their_budget(tmp_path):
    from services.exploration.pool import ExplorationPool, recover_interrupted
    root = tmp_path/'exploration'
    assert recover_interrupted(root) == []
    root.mkdir()
    rows = [{'episode': 1, 'env_id': 0, 'seed': 0, 'status': 'closed'},
            {'episode': 2, 'env_id': 1, 'seed': 1, 'status': 'active'}]
    (root/'state.json').write_text(json.dumps({'schema': 1, 'env_count': 2, 'seeds': [0, 1, 2],
                                               'episodes': rows, 'sealed': True}))
    with pytest.raises(RuntimeError, match='operator recovery'):
        ExplorationPool(root, count=2, seeds=[0, 1, 2], worker_config={})
    assert recover_interrupted(root) == [2]
    pool = ExplorationPool(root, count=2, seeds=[0, 1, 2], worker_config={})
    try:
        assert [r['status'] for r in pool.state['episodes']] == ['closed', 'interrupted']
        assert not pool.state['sealed'] and pool.status()['exploration_episodes_remaining'] == 2
        row = pool.reserve_rehearsal()  # The interrupted layout is played again.
        assert (row['episode'], row['seed'], row['seed_index']) == (3, 1, 1)
        with pytest.raises(RuntimeError, match='owner'):
            recover_interrupted(root)
    finally:
        pool._owner.close()


def test_resume_tells_the_agent_once_which_episodes_were_refunded(tmp_path):
    from services.exploration.pool import ExplorationPool, report_refunds
    root = tmp_path/'exploration'
    assert report_refunds(root) == ''
    root.mkdir()
    rows = [{'episode': 1, 'env_id': 0, 'seed': 0, 'status': 'closed'},
            {'episode': 2, 'env_id': 1, 'seed': 1, 'status': 'interrupted'},
            {'episode': 3, 'env_id': 0, 'seed': 2, 'status': 'error', 'error': 'RuntimeError: simulator exited (-9)'}]
    (root/'state.json').write_text(json.dumps({'schema': 1, 'env_count': 2, 'seeds': [0, 1, 2, 3],
                                               'episodes': rows, 'sealed': False}))
    before = (root/'state.json').read_text()
    note = report_refunds(root)
    assert 'episode 2 (interrupted: no error recorded); episode 3 (error: RuntimeError: simulator exited (-9))' in note
    assert 'episode 1 ' not in note and 'not charged' in note
    assert json.loads((root/'refunds-told.json').read_text()) == [2, 3]
    assert (root/'state.json').read_text() == before  # The pool's state is only read.
    assert report_refunds(root) == ''  # Told once.
    pool = ExplorationPool(root, count=2, seeds=[0, 1, 2, 3], worker_config={})
    try:
        assert pool.status()['exploration_episodes_refunded'] == 2
        # A live frontend owns the pool while the agent is relaunched; an episode that ends meanwhile
        # is told on the next turn, and episodes told before this file (refund_reported) stay told.
        state = json.loads((root/'state.json').read_text())
        state['episodes'] += [{'episode': 4, 'env_id': 1, 'seed': 3, 'status': 'error', 'error': 'RuntimeError: x'},
                              {'episode': 5, 'env_id': 0, 'seed': 0, 'status': 'interrupted', 'refund_reported': 1.0}]
        (root/'state.json').write_text(json.dumps(state))
        later = report_refunds(root)
        assert 'episode 4 (error: RuntimeError: x)' in later
        assert 'episode 5 ' not in later and 'episode 2 ' not in later and 'episode 3 ' not in later
        assert json.loads((root/'refunds-told.json').read_text()) == [2, 3, 4]
    finally:
        pool._owner.close()


def test_terminal_stop_is_what_the_dashboard_calls(tmp_path, monkeypatch):
    from harness import session_control
    from services.dashboard.sessions import ResearchSessions
    directory = tmp_path/'session'
    directory.mkdir()
    with pytest.raises(ValueError, match='No matching live launcher'):
        session_control.stop(directory)
    (directory/'agent-lifecycle.json').write_text(json.dumps({'pid': 1, 'status': 'running'}))
    with pytest.raises(ValueError, match='No matching live launcher'):
        session_control.stop(directory)  # PID 1 is never a launcher.
    assert ResearchSessions.alive is session_control.alive
    calls = []
    monkeypatch.setattr(session_control, 'stop', lambda d, is_alive: calls.append(d) or {'status': 'ok'})
    model = ResearchSessions(tmp_path/'project', lambda: [])
    monkeypatch.setattr(model, 'directory', lambda run_id: directory)
    assert model.stop('x') == {'status': 'ok'} and calls == [directory]


def test_sessions_have_no_wall_clock_or_model_call_cap():
    import inspect
    from harness.claude_cli import auto_research as claude
    for launch in (launcher.launch, claude.launch):
        assert not {'seconds', 'max_model_calls'} & set(inspect.signature(launch).parameters)
    assert 'seconds' not in inspect.signature(launcher.run_agent).parameters
    # One call's timeout stays inside the Claude CLI's Node timer range (2**31-1 ms).
    assert (launcher.CALL_SECONDS+300)*1000 < 2**31-1


# configuration.json and operator-input.json of a session prepared before agents lost their
# GPU (2026-10-11), field for field as configure_auto_research.py wrote them then; only the
# root, the image digest and the GPU UUIDs are placeholders.
LEGACY_CONFIGURATION = {
    'image': 'sha256:' + 'c'*64, 'task': TASK, 'exploration_seeds': [1, 2, 3, 4, 5],
    'formal_seed': 0, 'eval_seed': 0, 'sim_gpu': 'GPU-c4909b1f-facf-4d9f-049d-3d58cb2ca630',
    'controller_gpu': 'GPU-17fcc761-79d6-cba8-b2db-b8609a55c0c6',
    'training_gpu': 'GPU-17fcc761-79d6-cba8-b2db-b8609a55c0c6',
    'development': {'wall_seconds': 2400, 'memory_mb': 8192, 'cpus': 4, 'pids': 256, 'scratch_mb': 1024,
                    'artifact_bytes': 10737418240},
    'formal': {'wall_seconds': 2400, 'memory_mb': 8192, 'cpus': 4, 'pids': 256, 'scratch_mb': 1024,
               'artifact_bytes': 10737418240},
    'training': {'wall_seconds': 2400, 'memory_mb': 16384, 'cpus': 4, 'pids': 256, 'scratch_mb': 1024,
                 'artifact_bytes': 10737418240},
    'workspace_mb': 1048576, 'demonstration_context': 'terminal_state', 'formal_episodes': 50,
    'formal_eval_seed': 1, 'formal_workers': 4, 'observation_profile': 'rgbd', 'exploration_envs': 1,
    'subagents': 0, 'task_rules': 'public', 'agent_cli': 'codex', 'agent_provider': 'openrouter',
    'agent_effort': 'xhigh', 'agent_account': 'auto', 'agent_formal': False, 'write_handoff': False,
    'formal_context': 'compact', 'transfer_parts': ['memory', 'code', 'handoff'], 'transfer_mb': 100,
    'adaptation_source': None, 'adaptation_context': 'compact', 'benchmark': BENCH.name}


def frontend(root, environment, *requests):
    """The resumed launcher's MCP frontend (services/controller/frontend.py), as it starts it."""
    import subprocess
    import sys
    lines = ''.join(json.dumps(request) + '\n' for request in requests)
    return subprocess.run([sys.executable, '-m', 'services.controller.frontend',
                           '--config', str(root/'configuration.json'), '--workspace', str(root/'agent-workspace')],
                          input=lines, capture_output=True, text=True, timeout=120,
                          cwd=launcher.PROJECT, env=environment)


def test_a_session_saved_with_agent_gpus_resumes(tmp_path, monkeypatch):
    """A stopped session whose saved configuration still has controller_gpu and training_gpu
    resumes through the launcher, the frontend's supervisor (which refuses any change to the
    saved configuration) and the formal workers, all without an agent GPU."""
    import sys
    from dataclasses import asdict
    from services.controller.config import Configuration
    from services.controller.formal_worker import worker_configuration
    from services.controller.supervisor import Supervisor
    from test_controller_backend import Backend, Runner
    monkeypatch.setenv('RAX_ARTIFACT_ROOT', str(tmp_path))
    root = tmp_path/'auto-research'/'20261001T000000Z_0123456789'
    legacy = {'root': str(root), **LEGACY_CONFIGURATION}
    root.mkdir(parents=True)
    # The old launch: operator input and private configuration, then the supervisor's state
    # (written here by this supervisor, with the configuration the old one saved in it).
    for name in ('operator-input.json', 'configuration.json'):
        (root/name).write_text(json.dumps(legacy))
    Supervisor(Configuration.from_dict(legacy), lambda: Backend(), runner_factory=Runner).close()
    state = json.loads((root/'state.json').read_text())
    state['configuration'] = legacy
    state['exploration_started'] = 2
    (root/'state.json').write_text(json.dumps(state))
    (root/'agent-lifecycle.json').write_text(json.dumps({'pid': 1, 'started_at': 0, 'finished_at': 10,
                                                         'status': 'stopped'}))

    # 1. The launcher (--resume) loads operator-input.json; stop it where it would redeploy.
    class Redeployed(Exception):
        pass
    loaded = []
    def redeploy(config, codex=None):
        loaded.append(config)
        raise Redeployed
    monkeypatch.setattr(launcher, 'redeploy', redeploy)
    monkeypatch.setattr(launcher.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(sys, 'argv', ['auto-research', '--config', str(root/'operator-input.json'),
                                      '--resume', '--non-interactive'])
    with pytest.raises(Redeployed):
        launcher.main()
    config = loaded[0]
    assert (config.root, config.sim_gpu, config.task) == (root, legacy['sim_gpu'], TASK)
    assert not {'controller_gpu', 'training_gpu'} & set(asdict(config))

    # 2. The frontend loads configuration.json; its supervisor compares it with state.json.
    environment = {**os.environ, 'RAX_ARTIFACT_ROOT': str(tmp_path), 'PYTHONPATH': str(launcher.PROJECT)}
    environment.pop('OPENROUTER_API_KEY', None)
    status = {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call', 'params': {'name': 'status', 'arguments': {}}}
    result = frontend(root, environment, status)
    assert result.returncode == 0, result.stderr[-3000:]
    reply = json.loads(result.stdout.splitlines()[0])
    assert 'error' not in reply and not reply['result'].get('isError'), reply
    assert json.loads(reply['result']['content'][0]['text'])['exploration_remaining'] == 3
    saved = json.loads((root/'state.json').read_text())
    assert saved['configuration']['sim_gpu'] == legacy['sim_gpu'] and saved['exploration_started'] == 2
    # The same check refuses a real change, so the resume above passed it rather than skipped it.
    (root/'configuration.json').write_text(json.dumps({**legacy, 'formal_workers': 2}))
    refused = frontend(root, environment, status)
    assert refused.returncode != 0 and "A trial's saved configuration cannot change" in refused.stderr
    (root/'configuration.json').write_text(json.dumps(legacy))
    # Once a resumed supervisor saves its state, the retired fields are gone from state.json
    # but not from configuration.json; the session still resumes.
    resumed = Supervisor(Configuration.from_dict(legacy), lambda: Backend(), runner_factory=Runner)
    resumed._save()
    resumed.close()
    assert not {'controller_gpu', 'training_gpu'} & set(json.loads((root/'state.json').read_text())['configuration'])
    again = frontend(root, environment, status)
    assert again.returncode == 0, again.stderr[-3000:]

    # 3. Formal workers load the parent's configuration from state.json.
    parent = Configuration.from_dict(json.loads((root/'state.json').read_text())['configuration'])
    worker = worker_configuration(parent, 1)
    assert (worker.sim_gpu, worker.formal_seed) == (legacy['sim_gpu'], 1)
    assert not hasattr(worker, 'controller_gpu') and not hasattr(worker, 'training_gpu')


def test_claude_last_error_is_its_newest_api_error_whole(tmp_path):
    from harness.claude_cli import auto_research as claude
    assert claude.last_error(tmp_path) is None
    project = tmp_path/'projects'/'-workspace'
    project.mkdir(parents=True)
    error = 'API Error: 529 {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}'
    rows = [{'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'working'}]}},
            {'type': 'assistant', 'isApiErrorMessage': True, 'apiErrorStatus': 529,
             'message': {'content': [{'type': 'text', 'text': error}]}}]
    (project/'conv.jsonl').write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in rows))
    assert claude.last_error(tmp_path) == error


# configuration.json and operator-input.json of a session prepared before agents lost their
# GPU (2026-10-11), field for field as configure_auto_research.py wrote them then; only the
# root, the image digest and the GPU UUIDs are placeholders.
LEGACY_CONFIGURATION = {
    'image': 'sha256:' + 'c'*64, 'task': TASK, 'exploration_seeds': [1, 2, 3, 4, 5],
    'formal_seed': 0, 'eval_seed': 0, 'sim_gpu': 'GPU-c4909b1f-facf-4d9f-049d-3d58cb2ca630',
    'controller_gpu': 'GPU-17fcc761-79d6-cba8-b2db-b8609a55c0c6',
    'training_gpu': 'GPU-17fcc761-79d6-cba8-b2db-b8609a55c0c6',
    'development': {'wall_seconds': BENCH.task_timeouts(TASK)['episode_seconds'], 'memory_mb': 8192, 'cpus': 4, 'pids': 256, 'scratch_mb': 1024,
                    'artifact_bytes': 10737418240},
    'formal': {'wall_seconds': BENCH.task_timeouts(TASK)['episode_seconds'], 'memory_mb': 8192, 'cpus': 4, 'pids': 256, 'scratch_mb': 1024,
               'artifact_bytes': 10737418240},
    'training': {'wall_seconds': BENCH.task_timeouts(TASK)['episode_seconds'], 'memory_mb': 16384, 'cpus': 4, 'pids': 256, 'scratch_mb': 1024,
                 'artifact_bytes': 10737418240},
    'workspace_mb': 1048576, 'demonstration_context': 'none', 'expert_demonstrations': 0, 'expert_content': 'full',
    'formal_episodes': 50,
    'formal_eval_seed': 1, 'formal_workers': 4, 'observation_profile': 'rgbd', 'exploration_envs': 1,
    'subagents': 0, 'task_rules': 'public', 'agent_cli': 'codex', 'agent_provider': 'openrouter',
    'agent_effort': 'xhigh', 'agent_account': 'auto', 'agent_formal': False, 'write_handoff': False,
    'formal_context': 'compact', 'transfer_parts': ['memory', 'code', 'handoff'], 'transfer_mb': 100,
    'adaptation_source': None, 'adaptation_context': 'compact', 'benchmark': BENCH.name}


def frontend(root, environment, *requests):
    """The resumed launcher's MCP frontend (services/controller/frontend.py), as it starts it."""
    import subprocess
    import sys
    lines = ''.join(json.dumps(request) + '\n' for request in requests)
    return subprocess.run([sys.executable, '-m', 'services.controller.frontend',
                           '--config', str(root/'configuration.json'), '--workspace', str(root/'agent-workspace')],
                          input=lines, capture_output=True, text=True, timeout=120,
                          cwd=launcher.PROJECT, env=environment)


def test_a_session_saved_with_agent_gpus_resumes(tmp_path, monkeypatch):
    """A stopped session whose saved configuration still has controller_gpu, training_gpu and
    agent_account resumes through the launcher, the frontend's supervisor (which refuses any
    change to the saved configuration) and the formal workers, all without an agent GPU."""
    import sys
    from dataclasses import asdict
    from services.controller.config import Configuration
    from services.controller.formal_worker import worker_configuration
    from services.controller.supervisor import Supervisor
    from test_controller_backend import Backend, Runner
    monkeypatch.setenv('RAX_ARTIFACT_ROOT', str(tmp_path))
    root = tmp_path/'auto-research'/'20261001T000000Z_0123456789'
    legacy = {'root': str(root), **LEGACY_CONFIGURATION}
    root.mkdir(parents=True)
    # The old launch: operator input and private configuration, then the supervisor's state
    # (written here by this supervisor, with the configuration the old one saved in it).
    for name in ('operator-input.json', 'configuration.json'):
        (root/name).write_text(json.dumps(legacy))
    Supervisor(Configuration.from_dict(legacy), lambda: Backend(), runner_factory=Runner).close()
    state = json.loads((root/'state.json').read_text())
    state['configuration'] = legacy
    state['exploration_started'] = 2
    (root/'state.json').write_text(json.dumps(state))
    (root/'agent-lifecycle.json').write_text(json.dumps({'pid': 1, 'started_at': 0, 'finished_at': 10,
                                                         'status': 'stopped'}))

    # 1. The launcher (--resume) loads operator-input.json; stop it where it would redeploy.
    class Redeployed(Exception):
        pass
    loaded = []
    def redeploy(config, codex=None):
        loaded.append(config)
        raise Redeployed
    monkeypatch.setattr(launcher, 'redeploy', redeploy)
    monkeypatch.setattr(launcher.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(sys, 'argv', ['auto-research', '--config', str(root/'operator-input.json'),
                                      '--resume', '--non-interactive'])
    with pytest.raises(Redeployed):
        launcher.main()
    config = loaded[0]
    assert (config.root, config.sim_gpu, config.task) == (root, legacy['sim_gpu'], TASK)
    assert not {'controller_gpu', 'training_gpu', 'agent_account'} & set(asdict(config))

    # 2. The frontend loads configuration.json; its supervisor compares it with state.json.
    environment = {**os.environ, 'RAX_ARTIFACT_ROOT': str(tmp_path), 'PYTHONPATH': str(launcher.PROJECT)}
    environment.pop('OPENROUTER_API_KEY', None)
    status = {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call', 'params': {'name': 'status', 'arguments': {}}}
    result = frontend(root, environment, status)
    assert result.returncode == 0, result.stderr[-3000:]
    reply = json.loads(result.stdout.splitlines()[0])
    assert 'error' not in reply and not reply['result'].get('isError'), reply
    assert json.loads(reply['result']['content'][0]['text'])['exploration_remaining'] == 3
    saved = json.loads((root/'state.json').read_text())
    assert saved['configuration']['sim_gpu'] == legacy['sim_gpu'] and saved['exploration_started'] == 2
    # The same check refuses a real change, so the resume above passed it rather than skipped it.
    (root/'configuration.json').write_text(json.dumps({**legacy, 'formal_workers': 2}))
    refused = frontend(root, environment, status)
    assert refused.returncode != 0 and "A trial's saved configuration cannot change" in refused.stderr
    (root/'configuration.json').write_text(json.dumps(legacy))
    # Once a resumed supervisor saves its state, the retired fields are gone from state.json
    # but not from configuration.json; the session still resumes.
    resumed = Supervisor(Configuration.from_dict(legacy), lambda: Backend(), runner_factory=Runner)
    resumed._save()
    resumed.close()
    assert not {'controller_gpu', 'training_gpu', 'agent_account'} & set(
        json.loads((root/'state.json').read_text())['configuration'])
    again = frontend(root, environment, status)
    assert again.returncode == 0, again.stderr[-3000:]

    # 3. Formal workers load the parent's configuration from state.json.
    parent = Configuration.from_dict(json.loads((root/'state.json').read_text())['configuration'])
    worker = worker_configuration(parent, 1)
    assert (worker.sim_gpu, worker.formal_seed) == (legacy['sim_gpu'], 1)
    assert not any(hasattr(worker, key) for key in ('controller_gpu', 'training_gpu', 'agent_account'))


def launcher_main(tmp_path, monkeypatch, *options):
    """main() up to deployment, with a stand-in configuration rooted at tmp_path/'session'."""
    import types
    config = types.SimpleNamespace(root=tmp_path/'session', agent_cli='codex', agent_provider=None,
                                   adaptation_source=None, agent_formal=False)
    monkeypatch.setattr(launcher.Configuration, 'from_dict', lambda raw: config)
    monkeypatch.setattr('services.controller.config.with_task_timeouts', lambda c: c)
    monkeypatch.setattr('services.agent_formal.checkpoint.exploration_closed', lambda c: False)
    monkeypatch.setattr(launcher.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(launcher, 'redeploy', lambda c: pytest.fail('redeployed'))
    monkeypatch.setattr(launcher, 'deploy', lambda *a, **k: pytest.fail('deployed'))
    monkeypatch.setenv('HOME', str(tmp_path))
    source = tmp_path/'input.json'
    source.write_text('{}')
    monkeypatch.setattr(launcher.sys, 'argv', ['auto-research', '--config', str(source), *options])
    return config


def stopped_session(root):
    root.mkdir(parents=True)
    (root/'agent-lifecycle.json').write_text(json.dumps(
        {'pid': 1, 'started_at': time.time()-100, 'finished_at': time.time()-50, 'status': 'stopped'}))
    return (root/'agent-lifecycle.json').read_text()


@pytest.mark.parametrize('option', ['--resume', '--formal-only', '--agent-formal'])
def test_prepare_only_never_touches_an_existing_session(tmp_path, monkeypatch, capsys, option):
    """--resume --prepare-only once marked a stopped session prepared-only, which no resume accepted."""
    config = launcher_main(tmp_path, monkeypatch, option, '--prepare-only')
    before = stopped_session(config.root)
    with pytest.raises(SystemExit) as exit:
        launcher.main()
    assert exit.value.code == 2 and '--prepare-only checks a fresh deployment' in capsys.readouterr().err
    assert (config.root/'agent-lifecycle.json').read_text() == before
    assert launcher.resumable_lifecycle(config.root/'agent-lifecycle.json')[-1]['status'] == 'stopped'


def test_a_second_launcher_is_refused_with_a_message(tmp_path, monkeypatch):
    import fcntl
    config = launcher_main(tmp_path, monkeypatch, '--resume')
    before = stopped_session(config.root)
    with (config.root/'launch.lock').open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)  # The running launcher's lock.
        with pytest.raises(SystemExit, match=r'Another launcher is running this session .*stop_auto_research_agent'):
            launcher.main()
    assert (config.root/'agent-lifecycle.json').read_text() == before


def test_resuming_a_launch_that_failed_before_deployment_names_the_cause(tmp_path, monkeypatch):
    config = launcher_main(tmp_path, monkeypatch, '--resume')
    config.root.mkdir(parents=True)
    (config.root/'agent-lifecycle.json').write_text(json.dumps(
        {'pid': 1, 'started_at': 1, 'finished_at': 2, 'status': 'failed', 'error_type': 'ValueError'}))
    with pytest.raises(SystemExit) as exit:
        launcher.main()
    assert str(exit.value) == (
        'Nothing to resume: the first launch failed before the agent workspace was prepared (ValueError); read '
        f'its error in the launch output ({config.root/"agent.log"} for dashboard launches), fix the cause and '
        'prepare a fresh configuration')


@pytest.mark.parametrize('inherited', ['default', 'nohup'])
def test_a_closed_terminal_stops_the_launcher_unless_it_runs_under_nohup(tmp_path, monkeypatch, inherited):
    """The frontend runs in its own session, out of reach of a terminal's hangup: the launcher must
    turn SIGHUP into the same orderly stop as SIGTERM, or the frontend outlives it."""
    import signal
    launcher_main(tmp_path, monkeypatch, '--resume')
    installed = {}
    monkeypatch.setattr(launcher.signal, 'signal', lambda number, handler: installed.__setitem__(number, handler))
    monkeypatch.setattr(launcher.signal, 'getsignal', lambda number: signal.SIG_IGN if inherited == 'nohup'
                        and number == signal.SIGHUP else signal.SIG_DFL)
    with pytest.raises(SystemExit, match='never launched'):
        launcher.main()
    assert set(installed) == ({signal.SIGTERM} if inherited == 'nohup' else {signal.SIGTERM, signal.SIGHUP})
    for handler in installed.values():
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGHUP, None)
