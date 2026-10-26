"""Claude Code's custom Messages transport; real credentials stay on the host."""
from contextlib import contextmanager
from copy import deepcopy
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit


def prepare_request(body, model, effort):
    if body.get('model') != model or not isinstance(body.get('messages'), list):
        raise ValueError('Expected Messages for the configured model')
    body = deepcopy(body)
    body['output_config'] = {**body.get('output_config', {}), 'effort': effort}
    thinking = body.get('thinking', {})
    if thinking.get('type') == 'enabled' and 'budget_tokens' not in thinking:
        thinking['budget_tokens'] = body['max_tokens'] - 1
    # Some Messages gateways stringify image blocks nested in tool_result.
    # Keep the tool result association, but put its images in the user message.
    for message in body['messages']:
        if message.get('role') != 'user' or not isinstance(message.get('content'), list):
            continue
        lifted = []
        for block in message['content']:
            if block.get('type') != 'tool_result' or not isinstance(block.get('content'), list):
                continue
            images = [item for item in block['content'] if item.get('type') == 'image']
            if not images:
                continue
            block['content'] = [item for item in block['content'] if item.get('type') != 'image'] or [
                {'type': 'text', 'text': 'Tool images attached below.'}]
            lifted.extend([{'type': 'text', 'text': f"Images from tool result {block['tool_use_id']}:"}, *images])
        message['content'].extend(lifted)
    return body


def image_count(value):
    if isinstance(value, list):
        return sum(map(image_count, value))
    if isinstance(value, dict):
        return int(value.get('type') == 'image') + sum(map(image_count, value.values()))
    return 0


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *_):
        pass

    def do_POST(self):
        relay = self.server
        path = urlsplit(self.path).path
        if path not in ('/v1/messages', '/v1/messages/count_tokens'):
            self.send_error(403, 'Only Messages is enabled')
            return
        if self.headers.get('Authorization') != 'Bearer ' + relay.token:
            self.send_error(403, 'Invalid session credential')
            return
        upstream, started = None, False
        ident, status, usage = secrets.token_hex(12), None, {}
        returned_model, stop_reason, first_byte_seconds = None, None, None
        begin = time.time()
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 128 * 1024**2:
                raise ValueError('Invalid request size')
            incoming = json.loads(self.rfile.read(size))
            if path.endswith('/count_tokens'):
                if incoming.get('model') != relay.model:
                    raise ValueError('Unexpected model')
                body = prepare_request(incoming, relay.model, relay.effort)
                body.pop('output_config', None)
            else:
                body = prepare_request(incoming, relay.model, relay.effort)
            relay.record({'event': 'request', 'at': begin, 'request_id': ident, 'path': path,
                'model': body['model'], 'effort': body.get('output_config', {}).get('effort'),
                'thinking': body.get('thinking'), 'max_tokens': body.get('max_tokens'),
                'image_blocks': image_count(body.get('messages', [])),
                'tools': [t.get('name') for t in body.get('tools', [])]})
            url = urlsplit(relay.credential.base_url)
            cls = http.client.HTTPSConnection if url.scheme == 'https' else http.client.HTTPConnection
            upstream = cls(url.hostname, url.port, timeout=900)
            headers = {'Content-Type': 'application/json', **relay.credential.headers()}
            for name in ('anthropic-version', 'anthropic-beta'):
                if name in self.headers:
                    headers[name] = self.headers[name]
            upstream.request('POST', url.path.rstrip('/') + path,
                             body=json.dumps(body).encode(), headers=headers)
            response = upstream.getresponse()
            status = response.status
            self.send_response(status)
            self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
            self.end_headers()
            started, pending = True, b''
            while chunk := response.read1(65536):
                if first_byte_seconds is None:
                    first_byte_seconds = time.time() - begin
                self.wfile.write(chunk)
                self.wfile.flush()
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    if line.startswith(b'data:'):
                        try:
                            event = json.loads(line[5:])
                            message = event.get('message', {})
                            returned_model = message.get('model', returned_model)
                            stop_reason = event.get('delta', {}).get('stop_reason', stop_reason)
                            usage.update(message.get('usage', {}))
                            usage.update(event.get('usage', {}))
                        except (ValueError, AttributeError):
                            pass
                pending = pending[-4 * 1024**2:]
            if not usage and pending:
                try:
                    value = json.loads(pending)
                    usage = value.get('usage', {})
                    returned_model, stop_reason = value.get('model'), value.get('stop_reason')
                except (ValueError, AttributeError):
                    pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            if not started:
                self.send_error(502, 'Model transport failed: ' + type(error).__name__)
        finally:
            if upstream:
                upstream.close()
            relay.record({'event': 'response', 'at': time.time(), 'request_id': ident,
                'status': status, 'seconds': time.time() - begin, 'first_byte_seconds': first_byte_seconds,
                'model': returned_model, 'stop_reason': stop_reason, 'usage': usage})


@contextmanager
def serve(*, credential, model, effort, audit):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    server.credential, server.model, server.effort = credential, model, effort
    server.token = secrets.token_urlsafe(32)
    lock, audit = threading.Lock(), Path(audit)

    def record(event):
        with lock, audit.open('a') as log:
            log.write(json.dumps(event) + '\n')

    server.record = record
    server.base_url = f'http://127.0.0.1:{server.server_port}'
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
