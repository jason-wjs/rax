"""Host-owned robot/API MCP for the native agent CLI (Codex or Claude Code) and its Python."""
from contextlib import contextmanager
import argparse
import base64
import json
import os
from pathlib import Path, PurePosixPath
import signal
import selectors
import socket
import sys
import threading
import uuid

from .backend import make_backend
from harness import agent_host
from services.benchmarks import ALL_ROBOT_TOOLS, for_config as benchmark_for
from services.mcp_contract import Contract
from .errors import public_failure
from .config import Configuration, with_task_timeouts
from .gateway import Gateway, complete, sanitized
from .gemini import GeminiRouter, load_key, MAX_REQUEST_BYTES
from .recording import read_regular
from .sandbox import ExecutionDeadline, deadline
from .supervisor import Supervisor
from .storage import verify, project_directory
from services.transfer import PARTS, transfer_size


IMAGE_REPLY_GUIDANCE = (' Returns text/image content blocks. In functions.exec, iterate reply.content: '
                        'use text(block.text) for text blocks and image(block) for selected image blocks. '
                        'Never text(reply) or stringify image data.')
# The other CLIs display MCP image blocks directly; functions.exec is Codex's.
NATIVE_IMAGE_REPLY_GUIDANCE = (' Returns text/image content blocks; images appear directly in the tool result. '
                              'Never print or stringify image data.')


def for_cli(tools, cli):
    """Tool definitions with image guidance for the agent CLI that reads them."""
    if cli == 'codex':
        return tools
    return [{**tool, 'description': tool['description'].replace(IMAGE_REPLY_GUIDANCE, NATIVE_IMAGE_REPLY_GUIDANCE)}
            for tool in tools]


# Several exploration environments: development code binds one, the bundle never does.
REGISTER_MULTI_ENV = ('Freeze a controller.py bundle without running it. Define main(ctx) using the supplied '
    'single-environment context: ctx.call(...) without env_id, ctx.env(...), or a new Context(). Keep '
    'environment selection in a separate development wrapper outside the bundle.')
# Live-copy mode (agent formal): no bundle evaluation, so these tools read as practice.
LIVE_COPY_DESCRIPTIONS = {
    'status': ('Task, manual success evidence, episode state, episode timeout, remaining exploration episodes, '
               'workspace storage, Gemini budget, the carried folders and their sizes (carry), and once no '
               'episode remains and none is live, your last steps (next_step).'),
    'finish': 'Close the current exploration episode.',
}


def for_config(tools, config):
    """Tool definitions adjusted to this session's agent CLI and mode."""
    tools = for_cli(tools, config.agent_cli)
    if not benchmark_for(config).graded_score:
        tools = [{**tool, 'description': tool['description'].replace(GRADED_FORMAL, BINARY_FORMAL)}
                 if tool['name'] == 'submit' else tool for tool in tools]
    if not config.agent_formal:
        return tools
    return [{**tool, 'description': LIVE_COPY_DESCRIPTIONS.get(tool['name'], tool['description'])}
            for tool in tools]


# The submit description's scoring sentence, for graded and for binary (success/failure) benchmarks.
GRADED_FORMAL = 'Each formal episode earns the task score, with partial credit for partial progress; errors and missing evidence score zero.'
BINARY_FORMAL = 'Each formal episode succeeds or fails by the task condition; the result is the share of successes, and errors and missing evidence count as failures.'


def tool(name, description, properties=None, required=(), *, images=False):
    if images:
        description += IMAGE_REPLY_GUIDANCE
    return {'name': name, 'description': description, 'inputSchema': {'type': 'object',
            'properties': properties or {}, 'required': list(required), 'additionalProperties': False}}


STRING = {'type': 'string'}
SOURCE = {'source': {**STRING, 'description': 'Project under code/ in your workspace, e.g. code/reach. Rehearsals and the formal batch keep this path.'}}
DEFINITIONS = [
    tool('status', 'Task, manual success evidence, episode state, episode timeout, remaining exploration episodes, registered bundles (rehearsal_qualified: whether each bundle has had a clean rehearsal), workspace storage, and Gemini budgets: total and formal usage.'),
    tool('start_episode', 'Start the next exploration episode on the next layout; closes the previous one. Takes no layout or seed arguments.', images=True),
    tool('evaluate', 'Check the current exploration episode against the task; a success confirms manual success.'),
    tool('finish', 'Close the current exploration episode. Does not submit.', images=True),
    tool('register', 'Freeze a controller.py bundle without running it; main(ctx) must use only the ctx it is given.', SOURCE, ['source']),
    tool('rehearse', 'Run the whole registered controller in a fresh exploration episode under the same limits as each formal episode: its step limit and any wall limit (status.episode_timeout_seconds, startup included; null when there is none). Consumes one exploration episode. Returns wall-time breakdown in timing; timeouts identify timeout_stage.', {'bundle': STRING}, ['bundle'], images=True),
    tool('submit', 'One final submission, which closes exploration: freeze the exact cleanly rehearsed bundle for the formal batch (status.formal_episode_count held-out formal episodes); a clean rehearsal ran to its end without errors, whether or not the task succeeded. The formal batch runs it alone after you end your turn, each episode in a fresh sandbox and environment with no agent intervention, code changes or retries; you do not see its result. Each formal episode earns the task score, with partial credit for partial progress; errors and missing evidence score zero. The formal batch shares the Gemini budget.', {'bundle': STRING}, ['bundle']),
]


# The stage whose wall limit a lifecycle tool can exceed, named in its timeout error.
LIFECYCLE_DEADLINES = {'start_episode': 'environment startup', 'evaluate': 'evaluation',
                       'finish': 'evaluation'}

class ResearchFrontend:
    def __init__(self, supervisor, workspace, *, agent_pid_file=None):
        self.s = supervisor
        self.contract = Contract.from_config(supervisor.config)
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.agent_pid_file = agent_pid_file  # The running agent CLI (harness/agent_host.record).
        self.stopping = False  # Set by SIGTERM; no tool's cleanup may turn the stop into a tool error.
        self.robot_tools = self.contract.robot_definitions()
        bench = self.contract.bench
        class Inventory:
            """Episode-independent tools: Gemini and the benchmark's pure computations
            (its registry entry's unscoped_tools), no simulator."""
            def tools(_):
                return [t for t in self.robot_tools if t['name'] in bench.unscoped_tools]

            def call(_, name, arguments):
                if name not in bench.unscoped_tools:
                    raise RuntimeError('No interactive episode')
                return bench.unscoped_call(name, arguments), []
        self.api = Gateway(Inventory(), self.s.gemini, audit=self.s.audit, charge=self.s.charge,
                           refund=self.s.refund, budget=self.s.budget, provider_log=self.s.provider_log,
                           contract=Contract.from_config(self.s.config, isolated=True))
        self.api_token = self.api.acquire()

    def tools(self):
        return for_config(self.contract.select(self.robot_tools + DEFINITIONS + [
            t for t in self.api.definitions() if t['name'] == 'gemini_generate']), self.s.config)

    @contextmanager
    def frozen(self):
        # The agent's commands can leave background writers. Stop the agent CLI and
        # everything it runs while its submission is copied or executed.
        with agent_host.frozen(self.agent_pid_file):
            yield

    def source(self, name):
        path = self.workspace.joinpath(*project_directory(name).parts)
        # snapshot() pins every ancestor with O_NOFOLLOW as well.
        if not path.is_dir() or path.is_symlink():
            raise ValueError('Invalid source directory')
        return path

    def status(self):
        state = self.s.state
        storage = os.statvfs(self.workspace)
        demonstration_path = self.s.config.root/'demonstration-context.json'
        demonstration = json.loads(demonstration_path.read_text()) if demonstration_path.is_file() else {
            'kind': self.s.config.demonstration_context, 'manifest_path': None, 'image_count': 0}
        value = {'contract': self.contract.manifest(), 'task': self.s.config.task, 'manual_success_confirmed': bool(state['interactive_success']),
                'demonstration_context': demonstration,
                'workspace_storage': {'capacity_bytes': storage.f_blocks*storage.f_frsize,
                    'used_bytes': (storage.f_blocks-storage.f_bfree)*storage.f_frsize,
                    'available_bytes': storage.f_bavail*storage.f_frsize},
                'active_episode': (state.get('active') or {}).get('episode'),
                'episode_ended': bool(self.s.gateway and self.s.gateway.terminal),
                'exploration_remaining': len(self.s.config.exploration_seeds)-state['exploration_started'],
                # Rehearsal and each formal episode share this wall limit, startup included.
                'episode_timeout_seconds': self.s.config.formal.wall_seconds,
                'artifacts': self.s._recording.paths if self.s._recording else None,
                'api_budget': self.s.budget()}
        if self.contract.bundle_tools:
            value.update({
                'formal_reserved': state['formal_reserved'],
                'formal_episode_count': self.s.config.formal_episodes,
                'submitted_bundle': (state.get('submission') or {}).get('bundle'),
                'bundles': {ident: {'project_directory': str(project_directory(m['workspace_path'])),
                                    'sha256': m['sha256'],
                                    'rehearsal_qualified': self.s._has_qualifying_rehearsal(ident, m)}
                            for ident, m in state['bundles'].items()},
                # Formal usage counters and the same shared session dollar balance.
                'formal_api_budget': self.s.budget('formal')})
        return {**value, **self.closing_status()}

    def exploration_closed(self):
        """Budget spent with nothing running, or (code mode) the bundle already submitted."""
        state = self.s.state
        return bool(state['formal_reserved']) or (
            not state.get('active') and state['exploration_started'] >= len(self.s.config.exploration_seeds))

    def closing_status(self):
        """The carried folders (live-copy mode) and, once exploration is closed, the agent's
        last steps before it is frozen."""
        value = self.carry_status()
        if self.exploration_closed():
            marker = self.s.config.root/'exploration-complete.json'
            if not marker.exists():
                import time
                marker.write_text(json.dumps({'finished_at': time.time()}))
            value['next_step'] = self.next_step()
        return value

    def next_step(self):
        config = self.s.config
        if not config.agent_formal:
            state = self.s.state
            if state.get('submission'):
                return ('Exploration is complete. Update memory/PROGRESS.md, report your exploration results, '
                        'then end your turn. Your work is then frozen and the formal batch runs your submitted '
                        'bundle on the held-out layouts; you do not see its result.')
            if any(self.s._has_qualifying_rehearsal(ident, m) for ident, m in state['bundles'].items()):
                return ('No exploration episode remains and you have not submitted a bundle. Call submit with '
                        'the rehearsal-qualified bundle you choose (status lists them); without a submission '
                        'no formal batch runs. Then update memory/PROGRESS.md, report your exploration results '
                        'and end your turn.')
            return ('No exploration episode remains and no bundle has a qualifying rehearsal, so no formal '
                    'batch can run. Update memory/PROGRESS.md, report your exploration results, then end '
                    'your turn.')
        folders = ', '.join(PARTS[p] for p in config.transfer_parts)
        return ('No exploration episode remains. Update memory/PROGRESS.md and the carried folders '
                f'({folders}; at most {config.transfer_mb} MB in total), '
                + ('finish handoff/README.md as the transfer-handoff skill describes, '
                   if 'handoff' in config.transfer_parts else '')
                + 'copy into them any image you will need to see again, then end your turn. You continue '
                'from these folders in new episodes'
                + {'clear': '.', 'compact': ', with a summary of this conversation.',
                   'resume': ', with this whole conversation.'}[config.formal_context])

    def carry_status(self):
        """Live-copy mode: what a later copy of the agent carries, and how full it is."""
        config = self.s.config
        if not config.agent_formal:
            return {}
        sizes = transfer_size(self.workspace, config.transfer_parts)
        value = {'carried_folders': [PARTS[p] for p in config.transfer_parts],
                 'carried_bytes': sizes, 'carried_limit_bytes': config.transfer_mb * 1024**2,
                 'conversation': {'clear': 'none', 'compact': 'summary', 'resume': 'full'}[config.formal_context]}
        if sum(sizes.values()) > value['carried_limit_bytes']:
            value['warning'] = (f'The carried folders exceed {config.transfer_mb} MB. They are kept in the order '
                                'handoff, memory, code, and a folder that does not fit is left out whole; '
                                'trim them before you end your turn.')
        return {'carry': value}

    @staticmethod
    def public_result(result):
        return {k: result[k] for k in ('mode', 'reason', 'returncode', 'task_complete', 'episode',
            'episode_id', 'start_step_id', 'end_step_id', 'artifacts', 'recording_errors',
            'api_budget', 'simulator_exited', 'episode_ended', 'sha256', 'error_type',
            'error', 'start_failure', 'evaluation_error', 'evaluation_failure', 'timeout_seconds', 'timeout_stage', 'finalization_timeout', 'timing', 'output_error',
            'bundle', 'status', 'episode_count', 'completed_episodes', 'success_count',
            'unsuccessful_count', 'error_count', 'infrastructure_error_count', 'infrastructure_errors',
            'worker_returncode', 'success_rate', 'formal_episode_index', 'report_path',
            'formal_token_usage', 'token_usage_complete', 'formal_wall_seconds') if k in result}

    def robot(self, name, arguments):
        if self.s.state.get('formal_reserved'):
            raise RuntimeError('Exploration is closed to interactive control')
        gateway, token = self.s.gateway, self.s._interactive_token
        # Unscoped tools are pure computation: available with or without a live episode.
        if name in self.contract.bench.unscoped_tools or (name == 'gemini_generate' and gateway is None):
            gateway, token = self.api, self.api_token
        if gateway is None:
            raise RuntimeError('No interactive episode')
        response = gateway.handle(token, {'jsonrpc': '2.0', 'id': uuid.uuid4().hex,
            'method': 'tools/call', 'params': {'name': name, 'arguments': arguments}})
        if 'error' in response:
            return {'isError': True, 'content': [{'type': 'text', 'text': json.dumps({
                'error': response['error'],
                'artifacts': self.s._recording.paths if self.s._recording else None,
            })}]}
        return response['result']

    def call(self, name, args):
        if name in ALL_ROBOT_TOOLS or name == 'gemini_generate':
            self.contract.require_robot(name, args)
            return self.robot(name, args)
        definition = next((t for t in DEFINITIONS if t['name'] == name), None)
        if definition is None or name not in self.contract.tool_names:
            raise PermissionError('Unknown tool')
        schema = definition['inputSchema']
        if not isinstance(args, dict):
            raise ValueError('Invalid tool arguments')
        unexpected, missing = sorted(set(args)-set(schema['properties'])), sorted(set(schema['required'])-set(args))
        if unexpected or missing:
            raise ValueError('Invalid tool arguments' + (f'; unexpected: {", ".join(unexpected)}' if unexpected else '')
                             + (f'; missing: {", ".join(missing)}' if missing else ''))
        value = None
        if name == 'status':
            value = self.status()
        elif name == 'start_episode':
            if self.s.state['formal_reserved'] or self.s.state['exploration_started'] >= len(self.s.config.exploration_seeds):
                raise RuntimeError('No exploration episode remains; call status for the next step')
            if self.s.state['active']:
                self.s.finish_interactive()
            self.s.start_interactive()
            return self.robot(self.contract.bench.observe_tool, {})
        elif name == 'evaluate':
            if not self.s.gateway or self.s.state['formal_reserved']:
                raise RuntimeError('No interactive episode to evaluate')
            if self.s.gateway.simulator_exited:
                raise RuntimeError(f'Cannot evaluate: {self.s.gateway.simulator_exited}')
            with deadline(60):
                evaluation = self.s.backend.evaluate()
            if complete(evaluation):
                self.s._success()
            value = {'evaluation': sanitized(evaluation), **self.status()}
        elif name == 'finish':
            closed = None
            if self.s.state.get('active'):
                closed = self.public_result(self.s.finish_interactive())
            value = self.status()
            if closed:
                value['closed_episode'] = closed
                value['artifacts'] = closed.get('artifacts')
        elif name == 'register':
            with self.frozen():
                bundle, manifest = self.s.register(self.source(args['source']), workspace_path=args['source'])
            value = {'bundle': bundle, 'sha256': manifest['sha256'],
                     'project_directory': str(project_directory(manifest['workspace_path']))}
        elif name == 'submit':
            manifest = self.s.state['bundles'][args['bundle']]
            verify(self.s.config.root/'bundles'/manifest['directory'], manifest)
            if not self.s._has_qualifying_rehearsal(args['bundle'], manifest):
                raise RuntimeError('Exact clean rehearsal required')
            self.s.preflight(args['bundle'], formal=True, live_episode_ok=True)  # A failure leaves exploration open.
            if self.s.state.get('active'):
                self.s.finish_interactive()
            # The formal batch itself runs after the agent ends and is frozen (agent_formal.checkpoint).
            value = self.s.submit(args['bundle'])
            value.update(self.closing_status())  # Next step after submission.
        elif name == 'rehearse':
            manifest = self.s.state['bundles'][args['bundle']]
            verify(self.s.config.root/'bundles'/manifest['directory'], manifest)
            with self.frozen():
                if self.s.state.get('active'):
                    self.s.finish_interactive()
                value = self.public_result(self.s.run(args['bundle'], formal=False))
        return self.reply(value)

    def reply(self, value):
        """Small summary plus selected final frames."""
        result = {'content': [{'type': 'text', 'text': json.dumps(value, allow_nan=False)}]}
        if isinstance(value, dict) and value.get('reason') == 'timeout':
            result['isError'] = True
        artifacts = value.get('artifacts') if isinstance(value, dict) else None
        for image in (artifacts.get('final_images', []) if isinstance(artifacts, dict) else []):
            relative = PurePosixPath(image['path']).relative_to('runtime/autonomous_controller')
            if '..' in relative.parts:
                raise ValueError('Invalid image path')
            raw = read_regular(self.s.config.root/'published'/str(relative), 16*1024*1024)
            result['content'].append({'type': 'image', 'mimeType': 'image/png', 'data': base64.b64encode(raw).decode()})
        return result

    def close(self):
        self.api.revoke()
        self.s.close()

    def handle(self, request):
        if not isinstance(request, dict):
            return {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32600, 'message': 'Invalid request'}}
        ident = request.get('id')
        try:
            if request.get('jsonrpc') != '2.0':
                raise ValueError('Invalid JSON-RPC')
            method = request.get('method')
            if method == 'initialize':
                result = {'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}},
                          'serverInfo': {'name': 'auto-research', 'version': '3.0'}}
            elif method == 'notifications/initialized':
                return None
            elif method == 'tools/list':
                result = {'tools': self.tools()}
            elif method == 'tools/call':
                params = request['params']
                try:
                    result = self.call(params['name'], params.get('arguments', {}))
                except (Exception, ExecutionDeadline) as exc:
                    if isinstance(exc, ExecutionDeadline):
                        # A lifecycle wall limit (environment startup, evaluation) is a timeout of
                        # this call, not of the server: report it and keep serving.
                        exc = TimeoutError(f"{params.get('name')} did not finish within its wall-time limit"
                                           f" ({LIFECYCLE_DEADLINES.get(params.get('name'), 'lifecycle')}):"
                                           f" {exc or 'deadline reached'}")
                    failure = public_failure(exc, operation=params.get('name'), stage='lifecycle')
                    try:  # Robot calls audit their own errors; lifecycle errors are audited here.
                        self.s.audit({'event': 'mcp_error', 'tool': params.get('name'), **failure})
                    except Exception:
                        failure['audit_recording_failed'] = True
                    result = {'isError': True, 'content': [{'type': 'text',
                        'text': json.dumps({'error': failure})}]}
                if self.stopping and threading.current_thread() is threading.main_thread():
                    raise SystemExit(0)  # The stop arrived during this call; serve nothing further.
            else:
                raise PermissionError('Unsupported method')
            return {'jsonrpc': '2.0', 'id': ident, 'result': result}
        except Exception as exc:
            failure = public_failure(exc)
            return {'jsonrpc': '2.0', 'id': ident, 'error': {
                'code': -32000, 'message': failure['reason'], 'data': failure}}

    def serve_socket(self, listener):
        """Multiplex clients; execute every request on ONE main-thread dispatcher.

        An idle Codex connection must not block a Python client. Evaluation calls
        occupy this dispatcher, so development cannot act on their private episode.
        """
        listener.setblocking(False)
        with selectors.DefaultSelector() as poll:
            poll.register(listener, selectors.EVENT_READ)
            buffers = {}
            try:
                while True:
                    for key, _ in poll.select():
                        connection = key.fileobj
                        if connection is listener:
                            try:
                                client, _ = listener.accept()
                            except BlockingIOError:
                                continue
                            if len(buffers) >= 16:
                                client.close()
                                continue
                            client.setblocking(False)
                            buffers[client] = bytearray()
                            poll.register(client, selectors.EVENT_READ)
                            continue
                        try:
                            block = connection.recv(65536)
                            if not block:
                                raise ConnectionError('Client closed')
                            pending = buffers[connection]
                            pending.extend(block)
                            if len(pending) > MAX_REQUEST_BYTES:
                                raise ConnectionError('Request exceeds wire limit')
                            while b'\n' in pending:
                                line, _, tail = pending.partition(b'\n')
                                pending[:] = tail
                                try:
                                    request = json.loads(line)
                                except (ValueError, UnicodeError):
                                    response = {'jsonrpc': '2.0', 'id': None,
                                        'error': {'code': -32700, 'message': 'Invalid request'}}
                                else:
                                    response = self.handle(request)
                                if response is not None:
                                    connection.settimeout(5)
                                    connection.sendall(json.dumps(response, allow_nan=False).encode() + b'\n')
                                    connection.setblocking(False)
                        except (ConnectionError, OSError):
                            poll.unregister(connection)
                            buffers.pop(connection, None)
                            connection.close()
            finally:
                for client in buffers:
                    client.close()

    def serve(self, reader, writer):
        while line := reader.readline(MAX_REQUEST_BYTES + 1):
            if len(line.encode('utf-8')) > MAX_REQUEST_BYTES:
                writer.write(json.dumps({'jsonrpc': '2.0', 'id': None,
                    'error': {'code': -32600, 'message': 'Request exceeds wire limit'}}) + '\n')
                writer.flush()
                return  # Do not interpret the remaining fragment as another request.
            try:
                response = self.handle(json.loads(line))
            except Exception:
                response = {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'Invalid request'}}
            if response is not None:
                writer.write(json.dumps(response, allow_nan=False)+'\n')
                writer.flush()


def load_configuration(path):
    """The session's configuration with its task's current wall limits, as the launcher derives
    them on every start: a session saved under an older limit runs and
    resumes under the current one."""
    return with_task_timeouts(Configuration.from_dict(json.loads(Path(path).read_text())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--socket', type=Path)
    parser.add_argument('--agent-pid-file', type=Path)
    parser.add_argument('--key-file', type=Path)
    args = parser.parse_args()
    config = load_configuration(args.config)
    contract = Contract.from_config(config)
    if contract.environments > 1 and not args.socket:
        raise ValueError('Parallel exploration requires the shared socket endpoint')
    key = load_key(args.key_file) if config.external_perception else None
    supervisor = Supervisor(config, lambda: make_backend(config), GeminiRouter(key) if key else None)
    frontend_class = ResearchFrontend
    if config.exploration_envs > 1:
        from services.exploration.research import ParallelResearchFrontend
        frontend_class = ParallelResearchFrontend
    frontend = frontend_class(supervisor, args.workspace, agent_pid_file=args.agent_pid_file)
    reviewer = benchmark_for(config).episode_reviewer(config.root)
    if reviewer is not None:
        supervisor.episode_closed = reviewer.closed
        if getattr(frontend, 'pool', None) is not None:
            frontend.pool.episode_closed = reviewer.closed
        reviewer.start()
    def stop(*_):
        frontend.stopping = True
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    try:
        if args.socket:
            with socket.socket(socket.AF_UNIX) as listener:
                listener.bind(str(args.socket))
                os.chmod(args.socket, 0o600)
                listener.listen(16)
                frontend.serve_socket(listener)
        else:
            frontend.serve(sys.stdin, sys.stdout)
    finally:
        if reviewer:
            reviewer.close()
        frontend.close()


if __name__ == '__main__':
    main()
