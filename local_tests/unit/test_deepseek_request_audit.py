"""The DSH audit observes transport metadata without consuming or changing it."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

MODULE = Path(__file__).resolve().parents[2] / 'harness/deepseek_cli/request_audit.mjs'

SCRIPT = r"""
import { pathToFileURL } from 'node:url';
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
let input = ''; for await (const chunk of process.stdin) input += chunk;
const cfg = JSON.parse(input), encoded = new TextEncoder().encode(cfg.body ?? '');
let offset = 0, pulls = 0, cancelled = false, cancelReason, fetchArgs;
const failure = new Error('PRIVATE_ERROR_MESSAGE');
const source = new ReadableStream({
  pull(controller) {
    pulls++;
    if (cfg.stream_error && offset >= encoded.length) { controller.error(failure); return; }
    if (offset >= encoded.length) { controller.close(); return; }
    const end = Math.min(encoded.length, offset + (cfg.chunk_size ?? 7));
    controller.enqueue(encoded.slice(offset, end)); offset = end;
  },
  cancel(reason) { cancelled = true; cancelReason = reason; },
}, { highWaterMark: 0 });
globalThis.fetch = async function(...args) {
  fetchArgs = args;
  if (cfg.fetch_error) throw failure;
  const response = new Response(source, { status: cfg.status ?? 200, statusText: 'Provider status',
    headers: { 'content-type': cfg.content_type ?? 'text/event-stream; charset=utf-8', 'x-private-header': 'PRIVATE_HEADER' } });
  Object.defineProperties(response, { url: { value: 'https://provider.invalid/v1/messages' },
    redirected: { value: true }, clone: { value() { throw Error('Audit must not clone'); } } });
  return response;
};
process.env.RAX_DSH_AUDIT = join(process.argv[2], 'request-audit.jsonl');
await import(pathToFileURL(process.argv[1]).href);
const request = { model: 'requested-model', stream: true, thinking: { type: 'enabled' },
  output_config: { effort: 'max' }, tools: [{ name: 'robot_step' }],
  messages: [{ role: 'user', content: [{ type: 'text', text: 'PRIVATE_PROMPT' },
    { type: 'image', source: { type: 'base64', data: 'PRIVATE_IMAGE' } }] }] };
const init = { body: JSON.stringify(request), headers: { authorization: 'PRIVATE_KEY' } };
let result = {}, chunks = [];
try {
  const response = await fetch('https://provider.invalid/v1/' + (cfg.route ?? 'messages') + '?key=PRIVATE_QUERY', init);
  result = { status: response.status, statusText: response.statusText, url: response.url,
    redirected: response.redirected, header: response.headers.get('x-private-header'), pulls_before_read: pulls };
  const reader = response.body.getReader();
  if (cfg.cancel) {
    const value = await reader.read(); chunks.push(value.value);
    await reader.cancel('PRIVATE_CANCEL_REASON');
  } else {
    while (true) { const { value, done } = await reader.read(); if (done) break; chunks.push(value); }
  }
} catch (error) { result.error_preserved = error === failure; }
const readRows = name => { const path = join(process.argv[2], name);
  return existsSync(path) ? readFileSync(path, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse) : []; };
console.log(JSON.stringify({ ...result, cancelled, cancelReason, pulls,
  request_unchanged: fetchArgs?.[1] === init,
  bytes: Buffer.concat(chunks).toString('base64'),
  requests: readRows('request-audit.jsonl'), responses: readRows('response-audit.jsonl') }));
"""


def run_audit(tmp_path, **config):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is required for the native DSH transport')
    result = subprocess.run([node, '--input-type=module', '-e', SCRIPT, str(MODULE), str(tmp_path)],
                            input=json.dumps(config), capture_output=True, text=True, check=True, timeout=15)
    return json.loads(result.stdout)


START_USAGE = {'input_tokens': 113, 'output_tokens': 0, 'cache_read_input_tokens': 91,
               'cache_creation_input_tokens': 7,
               'cache_creation': {'ephemeral_5m_input_tokens': 7, 'ephemeral_1h_input_tokens': 0}}
END_USAGE = {'input_tokens': 113, 'output_tokens': 19, 'cache_read_input_tokens': 91}


def event(value, newline='\r\n'):
    return 'event: ' + value['type'] + newline + 'data: ' + json.dumps(value, ensure_ascii=False) + newline * 2


def full_body():
    return ': keepalive\r\n\r\n' + event({'type': 'message_start', 'message': {
        'model': 'returned-model', 'usage': START_USAGE}}) + event({'type': 'content_block_delta',
        'delta': {'type': 'text_delta', 'text': 'PRIVATE_RESPONSE中文🦾'}}) + event({
            'type': 'message_delta', 'delta': {'stop_reason': 'tool_use'}, 'usage': END_USAGE}) + event({'type': 'message_stop'})


@pytest.mark.parametrize('chunk_size', [1, 7, 65536])
def test_sse_metadata_chunk_boundaries_and_byte_preservation(tmp_path, chunk_size):
    import base64
    body = full_body()
    result = run_audit(tmp_path, body=body, chunk_size=chunk_size)
    assert base64.b64decode(result['bytes']) == body.encode()
    assert result['request_unchanged'] and result['pulls_before_read'] == 0
    assert (result['status'], result['statusText'], result['header'], result['redirected']) == (
        200, 'Provider status', 'PRIVATE_HEADER', True)
    assert result['url'] == 'https://provider.invalid/v1/messages'
    assert len(result['requests']) == 1  # Existing model-call accounting remains valid.
    request = result['requests'][0]
    assert request['model'] == 'requested-model'
    assert request['output_config'] == {'effort': 'max'}
    assert request['image_blocks'] == 1
    assert all(row['request_id'] == request['request_id'] for row in result['responses'])
    last = result['responses'][-1]
    assert last['event'] == 'response_end' and last['completed'] and not last['interrupted']
    assert last['returned_model'] == 'returned-model' and last['stop_reason'] == 'tool_use'
    assert last['raw_usage'] == {'message_start': START_USAGE, 'message_delta': END_USAGE}
    assert last['begin_at'] <= last['headers_at'] <= last['first_byte_at'] <= last['end_at']
    assert last['parse_status'] == 'parsed' and last['transport'] == 'eof'
    audit = json.dumps(result['requests'] + result['responses'])
    assert 'PRIVATE_' not in audit


def test_cancellation_propagates_without_draining_provider(tmp_path):
    result = run_audit(tmp_path, body=full_body(), chunk_size=1, cancel=True)
    assert result['cancelled'] and result['cancelReason'] == 'PRIVATE_CANCEL_REASON'
    assert result['pulls_before_read'] == 0 and result['pulls'] == 1
    assert result['responses'][-1]['transport'] == 'cancelled'
    assert result['responses'][-1]['interrupted'] and not result['responses'][-1]['completed']
    assert 'PRIVATE_' not in json.dumps(result['responses'])


def test_cancel_after_message_stop_preserves_model_completion(tmp_path):
    body = full_body()
    result = run_audit(tmp_path, body=body, chunk_size=len(body.encode()), cancel=True)
    assert result['cancelled'] and result['pulls'] == 1
    last = result['responses'][-1]
    assert last['transport'] == 'cancelled' and last['completed']
    assert not last['interrupted']
    assert last['returned_model'] == 'returned-model'
    assert last['raw_usage'] == {'message_start': START_USAGE, 'message_delta': END_USAGE}


@pytest.mark.parametrize('failure', ['fetch_error', 'stream_error'])
def test_errors_propagate_without_logging_error_text(tmp_path, failure):
    result = run_audit(tmp_path, body=event({'type': 'message_start', 'message': {
        'model': 'returned-model', 'usage': START_USAGE}}), **{failure: True})
    assert result['error_preserved']
    assert len(result['requests']) == 1
    assert result['responses'][-1]['transport'] == failure
    assert result['responses'][-1]['interrupted']
    assert 'PRIVATE_' not in json.dumps(result['responses'])


def test_eof_without_protocol_completion_is_interrupted(tmp_path):
    result = run_audit(tmp_path, body=event({'type': 'message_delta', 'delta': {
        'stop_reason': 'end_turn'}, 'usage': END_USAGE}))
    assert result['responses'][-1]['interrupted']
    assert not result['responses'][-1]['completed']
    assert result['responses'][-1]['raw_usage'] == {'message_delta': END_USAGE}


def test_oversized_content_does_not_hide_later_usage_or_change_stream(tmp_path):
    import base64
    body = event({'type': 'content_block_delta', 'delta': {'text': 'PRIVATE_RESPONSE' * 20000}}) + full_body()
    result = run_audit(tmp_path, body=body, chunk_size=65536)
    assert base64.b64decode(result['bytes']) == body.encode()
    last = result['responses'][-1]
    assert last['completed'] and last['parse_status'] == 'partial'
    assert last['oversized_events'] == 1 and last['raw_usage']['message_delta'] == END_USAGE
    assert len(json.dumps(result['responses'])) < 10000


@pytest.mark.parametrize('config,reason', [({'route': 'responses'}, 'unsupported_route'),
                                         ({'content_type': 'application/json'}, 'non_sse_content_type')])
def test_unsupported_protocols_are_explicitly_unparsed(tmp_path, config, reason):
    import base64
    result = run_audit(tmp_path, body='PRIVATE_RESPONSE', **config)
    assert base64.b64decode(result['bytes']) == b'PRIVATE_RESPONSE'
    last = result['responses'][-1]
    assert last['parse_status'] == 'unparsed' and last['unparsed_reason'] == reason
    assert last['completed'] is None and not last['interrupted']
    assert last['raw_usage'] == {} and 'PRIVATE_' not in json.dumps(result['responses'])


def test_http_and_provider_error_are_not_success(tmp_path):
    result = run_audit(tmp_path, status=429, body=event({'type': 'error', 'error': {
        'type': 'rate_limit_error', 'message': 'PRIVATE_ERROR_MESSAGE'}}))
    last = result['responses'][-1]
    assert last['status'] == 429 and last['provider_error']
    assert not last['completed'] and not last['interrupted']
    assert 'PRIVATE_' not in json.dumps(result['responses'])


def test_audit_write_failure_does_not_break_response(tmp_path):
    import base64
    result = run_audit(tmp_path / 'does-not-exist', body=full_body())
    assert base64.b64decode(result['bytes']) == full_body().encode()
    assert result['requests'] == result['responses'] == []
