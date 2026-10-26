"""The custom relay preserves Messages tools/images and holds the real API key."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from harness.claude_cli.login import Credential
from harness.claude_cli.messages_relay import prepare_request, serve
from test_bundle_sandbox import sandbox_runtime  # noqa: F401


def request_body():
    return {'model': 'deepseek-test', 'max_tokens': 1024, 'stream': True,
        'thinking': {'type': 'enabled'},
        'messages': [{'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'tool-1',
            'is_error': True, 'content': [{'type': 'text', 'text': 'partial observation'},
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'fixture-image'}}]}]}]}


def test_lifts_tool_images_without_changing_the_caller_or_tool_association():
    original = request_body()
    body = prepare_request(original, 'deepseek-test', 'max')
    content = body['messages'][0]['content']
    assert content[0] == {'type': 'tool_result', 'tool_use_id': 'tool-1', 'is_error': True,
                         'content': [{'type': 'text', 'text': 'partial observation'}]}
    assert content[1]['text'] == 'Images from tool result tool-1:'
    assert content[2] == original['messages'][0]['content'][0]['content'][1]
    assert len(original['messages'][0]['content'][0]['content']) == 2
    assert body['thinking']['budget_tokens'] == 1023
    assert body['output_config']['effort'] == 'max'
    with pytest.raises(ValueError, match='configured model'):
        prepare_request(original, 'other-model', 'max')


def test_custom_transport_streams_uses_host_key_and_logs_only_metadata(tmp_path):
    received = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            received.append((self.path, self.headers.get('Authorization'),
                             json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            data = b'data: {"type":"message_start","message":{"model":"deepseek-returned-revision","usage":{"input_tokens":21}}}\n\n' \
                   b'data: {"type":"message_delta","usage":{"output_tokens":3}}\n\n'
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    key = tmp_path/'private-key'
    key.write_text('host-only-fixture-key')
    key.chmod(0o600)
    credential = Credential('custom', key_file=key, base_url=f'http://127.0.0.1:{upstream.server_port}')
    audit = tmp_path/'audit.jsonl'
    try:
        with serve(credential=credential, model='deepseek-test', effort='max', audit=audit) as relay:
            headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + relay.token}
            with urlopen(Request(relay.base_url+'/v1/messages?beta=true',
                                 data=json.dumps(request_body()).encode(), headers=headers)) as response:
                result = response.read()
            assert b'message_delta' in result
            with pytest.raises(HTTPError) as denied:
                urlopen(Request(relay.base_url+'/v1/messages', data=b'{}'))
            assert denied.value.code == 403
            with pytest.raises(HTTPError) as denied:
                urlopen(Request(relay.base_url+'/other', data=b'{}', headers=headers))
            assert denied.value.code == 403
        assert received[0][0] == '/v1/messages'
        assert received[0][1] == 'Bearer host-only-fixture-key'
        assert received[0][2]['messages'][0]['content'][2]['type'] == 'image'
        rows = [json.loads(line) for line in audit.read_text().splitlines()]
        assert rows[0]['image_blocks'] == 1
        assert rows[-1]['usage'] == {'input_tokens': 21, 'output_tokens': 3}
        assert rows[-1]['model'] == 'deepseek-returned-revision'
        assert rows[-1]['first_byte_seconds'] >= 0
        assert 'host-only-fixture-key' not in audit.read_text()
        assert 'partial observation' not in audit.read_text() and 'fixture-image' not in audit.read_text()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)



def test_native_read_and_shell_cannot_reach_external_key_symlink(sandbox_runtime):
    import os
    import shlex
    import shutil
    from dataclasses import replace
    from test_auto_research_claude import sse
    from test_controller_backend import configuration
    from harness.claude_cli import auto_research as claude
    from services.controller.sandbox import deadline

    binary = shutil.which('claude')
    if not binary or not shutil.which('bwrap') or not shutil.which('socat'):
        pytest.skip('Native Claude sandbox is unavailable')
    config = replace(configuration(sandbox_runtime), agent_cli='claude', agent_provider='custom', agent_effort='max')
    secret = sandbox_runtime.parent / ('claude-external-key-' + str(os.getpid()))
    secret.write_text('fixture-external-credential-canary')
    secret.chmod(0o600)
    received = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            turns = sum(m['role'] == 'assistant' for m in body['messages'])
            if not any(t.get('name') == 'Bash' for t in body.get('tools', [])):
                blocks, stop = [{'type': 'text', 'text': 'ok'}], 'end_turn'
            else:
                received.append(body)
                if turns == 0:
                    blocks = [{'type': 'tool_use', 'id': 'read-canary', 'name': 'Read',
                               'input': {'file_path': str(config.root/'agent-workspace/key-alias')}}]
                elif turns == 1:
                    script = "from pathlib import Path; import os,json; r={}; " \
                             "r['real_key_in_environment']=os.environ.get('ANTHROPIC_AUTH_TOKEN')=='fixture-external-credential-canary'\n" \
                             "try: Path('key-alias').read_bytes(); r['key_readable']=True\n" \
                             "except OSError: r['key_readable']=False\n" \
                             "Path('key-boundary.json').write_text(json.dumps(r)); print('boundary saved')"
                    blocks = [{'type': 'tool_use', 'id': 'shell-canary', 'name': 'Bash',
                               'input': {'command': 'python -c ' + shlex.quote(script), 'description': 'check credential boundary'}}]
                else:
                    blocks = [{'type': 'text', 'text': 'done'}]
                stop = 'tool_use' if turns < 2 else 'end_turn'
            data = sse(blocks, stop, str(len(received)))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    try:
        deployed = claude.deploy(config, provider='custom', key_file=secret, model='deepseek-test', claude=binary,
                                base_url=f'http://127.0.0.1:{upstream.server_port}')
        (deployed['workspace']/'key-alias').symlink_to(secret)
        with deadline(120):
            result = claude.launch(config, deployed, prompt='Local boundary check. No robot episode.', capture=True)
        assert result.returncode == 0, result.stderr[-1000:]
        for request in received:
            advertised = {tool['name'] for tool in request.get('tools', [])}
            assert not advertised.intersection({'Agent', 'Task', 'Workflow', 'ScheduleWakeup', 'WebFetch', 'WebSearch'})
        results = [b for m in received[-1]['messages'] if isinstance(m['content'], list)
                   for b in m['content'] if b.get('type') == 'tool_result']
        assert results[0]['tool_use_id'] == 'read-canary' and results[0].get('is_error')
        assert 'fixture-external-credential-canary' not in json.dumps(results[0])
        boundary = json.loads((deployed['workspace']/'key-boundary.json').read_text())
        assert boundary == {'real_key_in_environment': False, 'key_readable': False}
    finally:
        upstream.shutdown()
        upstream.server_close()
        secret.unlink(missing_ok=True)
