"""Auto-research sessions: trusted launch controls and read-only monitoring.

Never instantiate the robot supervisor here: it owns an exclusive lock and live
environment state. Dashboard reads are snapshots, not another robot connection.
"""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time

import numpy as np
from PIL import Image

from services.codex_usage import CodexUsageReader
from services.controller.storage import persist
from services.controller.recording import read_regular
from services.controller.supervisor import qualifying_rehearsal
from services.controller.billing import NANODOLLARS, SESSION_LIMIT
from services.controller.config import FORMAL_EPISODES, MAX_EXPLORATION_EPISODES, MAX_FORMAL_EPISODES
from services.benchmarks import BENCHMARKS, DEFAULT, benchmark as benchmark_named, for_task
from services.scoring import FORMAL_WORKER_DIRECTORIES, operator_formal_report
from services.demonstration_io import DEMONSTRATION_CONTEXTS
from .observations import process_matches
from harness import session_control
from . import review

# Every registered benchmark camera, for the live view (a simulator lists its own in operator_state.json).
CAMERAS = tuple(dict.fromkeys(camera for bench in BENCHMARKS.values() for camera in bench.robot.get('cameras', ())))

RUN_ID = re.compile(r"^\d{8}T\d{6}Z_[a-f0-9]{10}$")
DEFAULTS = dict(task=benchmark_named(DEFAULT).task_names()[0], sim_gpu='0', episodes=5,
                observation_profile='rgbd', formal_seed=0, formal_episodes=FORMAL_EPISODES, formal_eval_seed=1, formal_workers=4,
                exploration_start_seed=1, exploration_envs=1, subagents=0,
                agent_formal='off', formal_context='compact', transfer_parts='memory,code,handoff',
                write_handoff='off', adaptation_source='', adaptation_context='compact',
                transfer_mb=100,
                demonstration_context='terminal_state', agent_cli='codex', agent_provider='openrouter', agent_effort='xhigh',
                task_rules='public', episode_reviews='off', expert_demonstrations=0, expert_content='full')
# A new form starts on the default benchmark's setup.
DEFAULTS.update(benchmark_named(DEFAULT).setup_defaults)


BOUNDS = dict(episodes=(2, MAX_EXPLORATION_EPISODES), formal_seed=(0, 1000000), formal_episodes=(1, MAX_FORMAL_EPISODES), formal_workers=(1, 16),
              exploration_envs=(1, 8), subagents=(0, 8), transfer_mb=(1, 1024),
              formal_eval_seed=(0, 1000000), exploration_start_seed=(0, 1000000), expert_demonstrations=(0, 3))


def read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _add_usage(total, usage):
    if usage:
        total = total or {}
        for key, value in usage.items():
            total[key] = total.get(key, 0) + value
    return total


def codex_usage(home, readers):
    """Token totals over every Codex thread of a session: the main (user) threads and the
    subagent threads, which Codex writes as separate rollout files."""
    totals = {'token_usage': None, 'main': None, 'subagents': None}
    threads = {'main': 0, 'subagents': 0}
    for rollout in sorted((home/'sessions').rglob('*.jsonl')):
        try:
            reader = readers.setdefault(str(rollout), CodexUsageReader())
            usage = reader.read(rollout)
            if not hasattr(reader, 'subagent'):
                with rollout.open() as stream:
                    meta = json.loads(stream.readline()).get('payload') or {}
                source = meta.get('source')
                reader.subagent = (meta.get('thread_source', 'user') != 'user'
                                   or isinstance(source, dict) and 'subagent' in source)
        except (OSError, ValueError):
            continue
        kind = 'subagents' if reader.subagent else 'main'
        threads[kind] += 1
        totals[kind] = _add_usage(totals[kind], usage['total_session_usage'])
        totals['token_usage'] = _add_usage(totals['token_usage'], usage['token_usage'])
    if not any(threads.values()):
        return None
    return {'token_usage': totals['token_usage'],
            'total_session_usage': _add_usage(_add_usage(None, totals['main']), totals['subagents']),
            'main_usage': totals['main'], 'subagent_usage': totals['subagents'], 'threads': threads}


def tail(path, maximum=65536):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-maximum))
            return stream.read(maximum).decode('utf-8', errors='replace')
    except OSError:
        return ''


def benchmark_of(task):
    """Task names are unique across benchmarks; each benchmark has its own catalog."""
    return for_task(task).name


def timeouts(task):
    return for_task(task).task_timeouts(task)


class ResearchSessions:
    def __init__(self, project, tasks, started_at=None):
        self.project = Path(project).resolve()
        self.root = (self.project/'runtime').resolve()/'auto-research'
        self.profile = (self.project/'runtime/operator/auto-research.json').resolve()
        self.tasks = tasks
        self.started_at = started_at
        self.lock = threading.RLock()
        self.usage_readers = {}
        self.observed_active = set()

    def directory(self, run_id, *, history=False):
        """A session directory. Sessions older than this server are review-only (history=True)."""
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise ValueError('Invalid auto-research session ID')
        path = self.root/run_id
        if not path.is_dir() or path.is_symlink() or path.resolve().parent != self.root.resolve():
            raise ValueError('Unknown auto-research session')
        created = datetime.strptime(run_id[:16], '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).timestamp()
        if not history and self.started_at is not None and created < self.started_at:
            if self.alive(path, read_json(path/'agent-lifecycle.json')):
                self.observed_active.add(run_id)
            elif run_id not in self.observed_active:
                raise ValueError('Session predates this dashboard server')
        return path

    def actionable(self, run_id):
        """Whether launch, resume or stop may act on this session (not review-only)."""
        try:
            self.directory(run_id)
            return True
        except ValueError:
            return False

    def settings(self, supplied):
        supplied = dict(supplied)
        supplied.pop('development_seconds', None)  # Ignore retired setting from older clients/profiles.
        supplied.pop('training_seconds', None)
        supplied.pop('formal_seconds', None)  # Retired: episode budgets follow the task's native steps.
        supplied.pop('agent_formal_max_model_calls', None)  # Retired: live copies have no call cap.
        supplied.pop('research_gpu', None)  # Retired: agents and their bundles have no GPU.
        supplied.pop('workspace_gib', None)  # Retired: agents run natively, without a capped workspace image.
        supplied.pop('agent_account', None)  # Retired: claude-login uses one experiment account.
        if set(supplied)-set(DEFAULTS):
            raise ValueError('Unknown auto-research setup field')
        values = {**DEFAULTS, **supplied}
        if values['task'] not in self.tasks():
            raise ValueError('Select an installed task')
        bench = for_task(values['task'])
        values['benchmark'] = bench.name
        # A source trial is a choice, not a default: refuse it rather than silently dropping it.
        if values['adaptation_source'] and 'adaptation_source' in bench.fixed:
            raise ValueError(bench.fixed['adaptation_source'][1])
        values = bench.apply_fixed(values)
        bench.validate_settings(values)
        from services.mcp_contract import Contract
        Contract(observation_profile=values['observation_profile'], environments=values['exploration_envs'],
                 subagents=values['subagents'], benchmark=values['benchmark'])
        if values['demonstration_context'] not in DEMONSTRATION_CONTEXTS:
            raise ValueError('Invalid demonstration_context')
        from services.controller.config import TASK_RULES
        if values['task_rules'] not in TASK_RULES:
            raise ValueError('Invalid task_rules')
        from harness.backends import NAMES, PROVIDERS, provider_choices
        if values['agent_cli'] not in NAMES:
            raise ValueError(f'agent_cli must be one of {NAMES}')
        providers = provider_choices(values['agent_cli'])
        if values['agent_provider'] not in providers:
            raise ValueError('Invalid agent_provider')
        if values['agent_cli'] in ('dsh', *PROVIDERS) and (values['agent_formal'] != 'off' or
                values['adaptation_source'] or values['subagents'] or values['exploration_envs'] != 1):
            raise ValueError(f"{values['agent_cli']} supports code formal, one environment, no adaptation or subagents")
        from services.controller.config import AGENT_EFFORTS
        if values['agent_effort'] not in AGENT_EFFORTS:
            raise ValueError('Invalid agent_effort')
        if values['agent_effort'] == 'high' and values['agent_cli'] not in ('codex', 'claude'):
            raise ValueError('high effort is currently supported only by Codex and Claude')
        from services.transfer import CONTEXTS, validate_parts
        if values['agent_formal'] not in ('off', 'on'):
            raise ValueError('agent_formal must be off or on')
        if values['formal_context'] not in CONTEXTS or values['adaptation_context'] not in CONTEXTS:
            raise ValueError('Invalid formal_context or adaptation_context')
        from services.controller.config import EXPERT_CONTENTS
        if values['expert_content'] not in EXPERT_CONTENTS:
            raise ValueError(f'expert_content must be one of {EXPERT_CONTENTS}')
        if values['episode_reviews'] not in ('off', 'on', 'detailed'):
            raise ValueError('episode_reviews must be off, on or detailed')
        if values['write_handoff'] not in ('off', 'on'):
            raise ValueError('write_handoff must be off or on')
        if values['adaptation_source']:
            # A finished trial of this dashboard, by run ID (often older than this server).
            # Kept as the run ID (the saved form profile reuses it); prepare() passes its path.
            source = self.directory(values['adaptation_source'], history=True)
            if not (source/'state.json').is_file():
                raise ValueError('Adaptation source has not run')
        if not isinstance(values['transfer_parts'], str):
            raise ValueError('transfer_parts must be comma-separated folder names')
        validate_parts(tuple(p for p in values['transfer_parts'].split(',') if p))
        for name, (low, high) in BOUNDS.items():
            if type(values[name]) is not int or not low <= values[name] <= high:
                raise ValueError(f'{name} must be an integer in {low}..{high}')
        if not isinstance(values['sim_gpu'], str) or not re.fullmatch(r'\d+|GPU-[a-fA-F0-9-]+', values['sim_gpu']):
            raise ValueError('Invalid sim_gpu')
        return values

    def prepare(self, supplied):
        values = self.settings(supplied)
        if values.get('agent_cli') == 'claude' and values.get('agent_provider') == 'claude-login':
            # Surface a missing or rejected experiment token before a session exists.
            from harness.claude_cli import experiment_login
            try:
                experiment_login.check()
            except (RuntimeError, OSError) as exc:
                raise ValueError(str(exc)) from None
        command = [str(self.project/'runtime/envs/host/bin/python'),
                   str(self.project/'scripts/configure_auto_research.py')]
        for key, value in values.items():
            if value == '' or value is None:
                continue  # Unset optional field (adaptation_source), or no wall limit (formal_seconds).
            if key == 'adaptation_source':
                value = self.directory(value, history=True)
            command += ['--'+key.replace('_', '-'), str(value)]
        with self.lock:
            result = subprocess.run(command, cwd=self.project, capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise ValueError('Configuration/preflight failed: '+result.stderr[-2000:])
            path = Path(result.stdout.strip())
            directory = self.directory(path.parent.name)
            if path != directory/'operator-input.json' or not path.is_file():
                raise ValueError('Configuration command returned an unexpected path')
            self.profile.parent.mkdir(parents=True, exist_ok=True)
            persist(self.profile, values)
            return self.detail(directory.name)

    def command(self, directory):
        return 'cd '+shlex.quote(str(self.project))+' && '+shlex.join([
            'bash', 'scripts/start_auto_research_agent.sh', '--config', str(directory/'operator-input.json')])

    def launch(self, run_id):
        with self.lock:
            directory = self.directory(run_id)
            if any((directory/name).exists() for name in ('launch-request.json', 'agent-lifecycle.json', 'agent-workspace', 'state.json')):
                raise ValueError('Session was already launched; prepare a fresh configuration')
            if not (directory/'operator-input.json').is_file():
                raise ValueError('No prepared configuration')
            persist(directory/'launch-request.json', {'requested_at': time.time()})
            with (directory/'agent.log').open('ab') as log:
                process = subprocess.Popen(['bash', str(self.project/'scripts/start_auto_research_agent.sh'),
                    '--config', str(directory/'operator-input.json'), '--non-interactive'],
                    cwd=self.project, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            threading.Thread(target=process.wait, daemon=True).start()  # Reap without owning its lifetime.
            return {'run_id': run_id, 'pid': process.pid, 'status': 'launching'}

    def resume(self, run_id):
        """Relaunch a stopped, exited, failed or crashed session, continuing its own conversation."""
        with self.lock:
            directory = self.directory(run_id)
            lifecycle = read_json(directory/'agent-lifecycle.json')
            if not self.resumable(directory, lifecycle, read_json(directory/'operator-input.json')):
                raise ValueError('Only a launched auto-research session whose launcher has stopped can be resumed')
            with (directory/'agent.log').open('ab') as log:
                process = subprocess.Popen(['bash', str(self.project/'scripts/start_auto_research_agent.sh'),
                    '--config', str(directory/'operator-input.json'), '--non-interactive', '--resume'],
                    cwd=self.project, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            threading.Thread(target=process.wait, daemon=True).start()
            return {'run_id': run_id, 'pid': process.pid, 'status': 'resuming'}

    def resumable(self, directory, lifecycle, config):
        return (bool(lifecycle) and lifecycle.get('status') != 'prepared-only' and not lifecycle.get('kind')
                and (directory/'agent-workspace').exists() and not self.alive(directory, lifecycle))

    alive = staticmethod(session_control.alive)

    def stop(self, run_id):
        """The terminal's scripts/stop_auto_research_agent.sh, for this run."""
        return session_control.stop(self.directory(run_id), is_alive=self.alive)

    def summary(self, directory):
        config = read_json(directory/'operator-input.json')
        state = read_json(directory/'state.json')
        lifecycle = read_json(directory/'agent-lifecycle.json')
        alive = self.alive(directory, lifecycle)
        status = lifecycle.get('status', 'launching' if (directory/'launch-request.json').exists() else 'prepared')
        # Statuses of a live launcher; with its launcher gone, the run was interrupted.
        if status in ('preparing', 'running', 'usage_wait', 'formal') and not alive:
            status = 'interrupted'
        active = state.get('active') or {}
        results = state.get('results', [])
        rehearsed = [bundle for bundle, manifest in state.get('bundles', {}).items()
                     if any(qualifying_rehearsal(result, bundle, manifest) for result in results+state.get('imported_rehearsals', []))]
        bundle_rows = [r for r in results if r.get('mode') == 'formal' and r.get('driver', 'bundle') == 'bundle']
        formal = bundle_rows[-1] if bundle_rows else None
        batch = operator_formal_report(directory, state.get('formal_batch'), bundle_rows)
        agent_batch = operator_formal_report(directory, state.get('agent_formal_batch'),
                                            [r for r in results if r.get('driver') == 'agent'])
        if batch:
            batch_status = batch['status'] if batch['status'] != 'running' or alive else 'interrupted'
            stage = f"Formal batch {batch_status}: {batch['completed_episodes']}/{batch['episode_count']} completed, {batch['success_count']} successes"
            if batch.get('success_rate') is not None:
                stage += f" ({100*batch['success_rate']:.1f}%)"
        elif formal:
            stage = 'Formal complete' if formal.get('task_complete') is True else 'Formal ended without confirmed success'
        elif state.get('formal_reserved'):
            stage = ('Formal running' if alive and active.get('mode') == 'formal'
                     else 'Submitted; formal runs once the agent is frozen' if state.get('submission') and alive
                     else 'Formal reserved; no retry')
        elif active.get('mode') == 'rehearsal':
            stage = 'Rehearsal running' if alive else 'Rehearsal interrupted'
        elif rehearsed:
            stage = 'Qualifying rehearsal recorded'
        elif state.get('interactive_success'):
            stage = 'Autonomous development / rehearsal'
        else:
            stage = 'Direct robot control' if state else 'Not started'
        if agent_batch:
            agent_status = agent_batch['status'] if agent_batch['status'] != 'running' or alive else 'interrupted'
            stage += (f"; agent formal {agent_status}: {agent_batch['completed_episodes']}/{agent_batch['episode_count']}"
                      f" completed, {agent_batch['success_count']} successes")
        elif lifecycle.get('status') == 'formal' and config.get('agent_formal'):
            stage += '; frozen, agent formal pending'
        spend = state.get('gemini_spend', {})
        reported, reserved = spend.get('reported', 0)/NANODOLLARS, spend.get('reserved', 0)/NANODOLLARS
        limit = None if SESSION_LIMIT is None else SESSION_LIMIT/NANODOLLARS
        return dict(run_id=directory.name, task=config.get('task'), status=status, alive=alive, stage=stage,
            demonstration_context=config.get('demonstration_context', 'none'),
            sim_gpu=config.get('sim_gpu'),
            exploration_used=state.get('exploration_started', 0), exploration_limit=len(config.get('exploration_seeds', [])),
            exploration_pool=state.get('exploration_pool'),
            manual_success=bool(state.get('interactive_success')), formal_reserved=bool(state.get('formal_reserved')),
            active=active, formal_batch=batch, agent_formal_batch=agent_batch,
            agent_checkpoint=state.get('agent_checkpoint'), rehearsed_bundles=rehearsed, gemini=dict(limit_usd=limit, reported_usd=reported, reserved_usd=reserved,
                remaining_usd=None if limit is None else max(0, limit-reported-reserved), blocked=spend.get('blocked', False)),
            agent=self.agent(directory, config),
            launchable=status == 'prepared' and not (directory/'agent-workspace').exists(),
            resumable=self.resumable(directory, lifecycle, config), attempts=len(lifecycle.get('attempts', [])) + bool(lifecycle))

    @staticmethod
    def agent(directory, config):
        """Agent CLI and provider; for claude-login, the dedicated account it bills."""
        cli, provider = config.get('agent_cli', 'codex'), config.get('agent_provider', 'openrouter')
        result = {'cli': cli, 'provider': provider if cli != 'codex' else None, 'account': None,
                  'effort': config.get('agent_effort', 'xhigh')}
        if cli == 'claude' and provider == 'claude-login':
            login = read_json(directory/'agent-login.json')  # Written at deploy; identity only.
            if login:
                result['account'] = (login.get('account') or {}).get('email')
            else:
                from harness.claude_cli import experiment_login
                result['account'] = experiment_login.identity().get('email')
                result['pending'] = True  # Recorded when the session launches.
        return result

    def detail(self, run_id, *, history=False):
        directory = self.directory(run_id, history=history)
        summary = self.summary(directory)
        if history and summary['resumable'] and not self.actionable(run_id):
            # Review-only (older than this server and never seen running): resume() refuses it.
            summary['resumable'] = False
        state = read_json(directory/'state.json')
        # Codex sessions: every rollout thread; Claude sessions: the transcripts in their home.
        usage = codex_usage(directory/'codex-home', self.usage_readers)
        if usage is None and (directory/'claude-home').is_dir():
            from services.claude_usage import UsageReader
            usage = self.usage_readers.setdefault(str(directory/'claude-home'), UsageReader()).read(
                directory/'claude-home')
        if (directory/'dsh-home').is_dir():
            from services.dsh_usage import UsageReader
            usage = UsageReader().read(directory/'dsh-home')
        from harness.backends import PROVIDERS
        for name in PROVIDERS:
            if (directory/(name+'-home')).is_dir():
                if name == 'cursor':
                    from services.cursor_usage import UsageReader
                else:
                    from services.native_cli_usage import UsageReader
                usage = UsageReader().read(directory/(name+'-home'))
        return {**summary, 'command': self.command(directory),
                'live_observation': self.live_observation(directory),
                'configuration': read_json(directory/'operator-input.json'), 'usage': usage,
                'subagents': read_json(directory/'subagents.json') or None,
                'results': state.get('results', [])[-100:], 'bundles': state.get('bundles', {}),
                'workspace': str(directory/'agent-workspace'), 'artifacts': str(directory/'published'),
                'agent_log': tail(directory/'agent.log'), 'frontend_log': tail(directory/'frontend.log')}

    @staticmethod
    def latest_native(directory):
        """Newest simulator state; during a concurrent formal batch, the lowest live episode.

        Preferring a stable live worker keeps the view from jumping between episodes.
        """
        candidates = []
        paths = [*(directory/'native/results').glob('*/sim/operator_state.json'),
                 *directory.glob('exploration/episode-*/native/results/*/sim/operator_state.json'),
                 *(path for workers in FORMAL_WORKER_DIRECTORIES
                   for path in directory.glob(f'{workers}/episode-*/native/results/*/sim/operator_state.json'))]
        for path in paths:
            try:
                state = json.loads(read_regular(path, 65536))
                if not isinstance(state, dict):
                    continue
                candidates.append((path.stat().st_mtime_ns, str(path), path.parent, state))
            except (OSError, ValueError):
                continue
        if not candidates:
            return None, None
        live = sorted((c for c in candidates if any(f'/{workers}/' in c[1] for workers in FORMAL_WORKER_DIRECTORIES)
                       and process_matches(c[3].get('simulator_pid'), c[2])), key=lambda c: c[1])
        _, _, sim, state = live[0] if live else max(candidates, key=lambda entry: entry[:2])
        return sim, state

    @staticmethod
    def observation_path(sim, state):
        relative = Path(state.get('observation_file') or '')
        if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.npz':
            raise ValueError('Invalid native observation path')
        return sim/relative

    def live_observation(self, directory):
        sim, state = self.latest_native(directory)
        if state is None:
            return None
        cameras = [camera for camera in state.get('cameras', []) if camera in CAMERAS]
        try:
            available = self.observation_path(sim, state).is_file()
        except ValueError:
            available = False
        return {'native_run': sim.parent.name, 'episode_id': state.get('episode_id'),
                'episode_mode': state.get('episode_mode'), 'step_id': state.get('step_id'),
                'episode_step_limit': state.get('episode_step_limit'),
                'active': process_matches(state.get('simulator_pid'), sim),
                'updated_at_unix_s': state.get('updated_at_unix_s'),
                'available': available, 'cameras': cameras if available else [],
                'reward': state.get('reward', {}), 'reason': state.get('reason')}

    def frame(self, run_id, camera, native_run, step_id, *, history=False):
        directory = self.directory(run_id, history=history)
        if camera not in CAMERAS:
            raise ValueError('Unknown camera')
        sim, state = self.latest_native(directory)
        if state is None:
            raise FileNotFoundError('No native observation yet')
        # A reset/step between metadata and image fetch must not mix episodes/frames.
        if sim.parent.name != native_run or str(state.get('step_id')) != str(step_id):
            raise FileNotFoundError('Observation changed; refresh the live view')
        raw = read_regular(self.observation_path(sim, state), 64*1024*1024)
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if camera not in archive:
                raise FileNotFoundError('Camera is unavailable')
            rgb = archive[camera]
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
                raise ValueError('Invalid RGB observation')
            stream = io.BytesIO()
            Image.fromarray(rgb[..., :3]).save(stream, format='JPEG', quality=88, optimize=True)
        return stream.getvalue()

    def artifact(self, run_id, logical_path, *, history=False):
        directory = self.directory(run_id, history=history)
        relative = Path(logical_path)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[:3] != ('runtime', 'autonomous_controller', 'runs'):
            raise ValueError('Only published run artifacts are available')
        path = directory/'published'/Path(*relative.parts[2:])
        if path.suffix not in {'.png', '.json', '.jsonl', '.log', '.txt'}:
            raise ValueError('Select a PNG image or text artifact')
        # Reject links/special files, and bound responses before allocation.
        data = read_regular(path, 4*1024*1024)
        return data, 'image/png' if path.suffix == '.png' else 'text/plain; charset=utf-8'

    def episodes(self, run_id, *, history=False):
        directory = self.directory(run_id, history=history)
        from services.episode_reviews import runner as reviews
        records = reviews.records(directory)
        rows = review.episodes(directory)
        for row in rows:
            record = records.get(row.get('episode_id') or '')
            row['review'] = {'status': record.get('status'), 'cost_usd': record.get('cost_usd')} if record else None
        return rows

    def reviews(self, run_id, *, history=False):
        """Episode-review setting and spend of a session (benchmarks with episode reviews)."""
        directory = self.directory(run_id, history=history)
        from services.episode_reviews.common import SESSION_BUDGET_USD
        from services.episode_reviews import runner as reviews
        benchmark = benchmark_named(read_json(directory/'configuration.json').get('benchmark', DEFAULT))
        return {**reviews.settings(directory), 'available': benchmark.episode_reviews,
                'spent_usd': round(reviews.spent(directory), 4), 'budget_usd': SESSION_BUDGET_USD}

    def episode_review(self, run_id, episode_id, *, history=False):
        from services.episode_reviews import runner as reviews
        path = reviews.record_path(self.directory(run_id, history=history), episode_id)
        record = read_json(path)
        if not record:
            raise FileNotFoundError('No review for this episode')
        return record

    def video(self, run_id, episode, *, history=False):
        return review.video_path(self.directory(run_id, history=history), episode)

    def status(self, run_id=None, *, history=False):
        sessions = []
        for directory in sorted(self.root.glob('*'), key=lambda p: p.name, reverse=True):
            try:
                self.directory(directory.name, history=history)
            except ValueError:
                continue
            if (directory/'operator-input.json').is_file():
                sessions.append(self.summary(directory))
        selected = run_id or (sessions[0]['run_id'] if sessions else None)
        saved = read_json(self.profile)
        profile = {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}
        detail = self.detail(selected, history=history) if selected else None
        if detail:
            # Launch/stop never act on sessions older than this server.
            detail['launchable'] = detail['launchable'] and not history
        from harness.backends import NAMES, provider_choices
        return {'sessions': sessions, 'selected': detail,
                'agent_providers': {name: list(provider_choices(name)) for name in NAMES},
                'profile': profile, 'task_timeouts': {t: timeouts(t) for t in self.tasks()},
                'tasks': self.tasks(), 'demonstration_contexts': list(DEMONSTRATION_CONTEXTS),
                'benchmarks': {t: benchmark_of(t) for t in self.tasks()},
                # Form values applied when the selected task switches benchmark (still editable).
                'benchmark_defaults': {name: b.setup_defaults for name, b in BENCHMARKS.items()},
                # Fields that do not apply to a benchmark: the form disables them and the server fixes them.
                'benchmark_fixed': {name: {k: {'value': v, 'reason': r} for k, (v, r) in b.fixed.items()}
                                    for name, b in BENCHMARKS.items()},
                # What the page shows per benchmark, so it never branches on benchmark names.
                'benchmark_info': {name: {'label': b.label, 'task_tag': b.task_tag, 'score_label': b.score_label,
                                          'layout_note': b.layout_note} for name, b in BENCHMARKS.items()}}
