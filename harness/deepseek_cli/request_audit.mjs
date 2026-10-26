// Metadata only: never record headers, prompts, images, or generated content.
import { appendFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { dirname, join } from 'node:path';

const original = globalThis.fetch;
const now = () => Date.now() / 1000;
function append(path, record) {
  try { appendFileSync(path, JSON.stringify(record) + '\n', { mode: 0o600 }); }
  catch { /* Observability must never change provider traffic. */ }
}
function imageCount(value) {
  if (!value || typeof value !== 'object') return 0;
  if (Array.isArray(value)) return value.reduce((sum, item) => sum + imageCount(item), 0);
  if (['image_url', 'image', 'input_image'].includes(value.type)) return 1;
  return Object.values(value).reduce((sum, item) => sum + imageCount(item), 0);
}

// Bound each SSE event, including ignored content events. Decode in small slices
// so a single unusually large fetch chunk cannot create an unbounded text buffer.
function messagesParser(onMetadata) {
  const decoder = new TextDecoder();
  const limit = 64 * 1024;
  let line = '', data = [], size = 0, dropping = false, afterCR = false;
  const stats = { oversized_events: 0, malformed_events: 0 };
  function dispatch() {
    if (dropping) stats.oversized_events++;
    else if (data.length) {
      try {
        const value = JSON.parse(data.join('\n'));
        if (['message_start', 'message_delta', 'message_stop', 'error'].includes(value.type)) onMetadata(value);
      } catch { stats.malformed_events++; }
    }
    data = []; size = 0; dropping = false;
  }
  function endLine() {
    if (!line) dispatch();
    else if (!dropping && line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    line = '';
  }
  function decode(text) {
    for (const char of text) {
      if (afterCR) { afterCR = false; if (char === '\n') continue; }
      if (char === '\r' || char === '\n') {
        endLine(); afterCR = char === '\r';
      } else {
        // Keep a sentinel for a dropped nonempty line; its newline is not an
        // event delimiter. UTF-16 length is bounded, independently of UTF-8.
        if (!dropping && ++size > limit) { dropping = true; data = []; line = 'x'; }
        else if (!dropping) line += char;
        else if (!line) line = 'x';
      }
    }
  }
  return {
    stats,
    write(bytes) {
      for (let offset = 0; offset < bytes.byteLength; offset += 16384) {
        decode(decoder.decode(bytes.subarray(offset, offset + 16384), { stream: true }));
      }
    },
    finish() {
      decode(decoder.decode());
      // SSE events require a blank line; do not accept a truncated final event.
      if (dropping) stats.oversized_events++;
      else if (line || data.length) stats.malformed_events++;
    },
  };
}

globalThis.fetch = async function(input, init) {
  const auditPath = process.env.RAX_DSH_AUDIT;
  let url;
  try { url = new URL(String(input?.url ?? input)); } catch { /* Not an audited endpoint. */ }
  if (!auditPath || !url || !/\/(chat\/completions|responses|messages)$/.test(url.pathname)) {
    return Reflect.apply(original, this, [input, init]);
  }
  url.username = ''; url.password = ''; url.search = ''; url.hash = '';
  const requestId = randomUUID(), begin = now(), started = performance.now();
  const responsePath = join(dirname(auditPath), 'response-audit.jsonl');
  const request = { at: begin, request_id: requestId, endpoint: url.href };
  try {
    const body = JSON.parse(init?.body ?? '{}');
    Object.assign(request, {
      model: body.model, reasoning: body.reasoning, reasoning_effort: body.reasoning_effort,
      thinking: body.thinking, output_config: body.output_config,
      max_tokens: body.max_tokens ?? body.max_completion_tokens ?? body.max_output_tokens,
      tools: (body.tools ?? []).map(tool => tool.function?.name ?? tool.name),
      image_blocks: imageCount(body.messages ?? body.input ?? []),
    });
  } catch { request.request_body_unparsed = true; }
  // Keep exactly one request line: the existing DSH usage reader counts these.
  append(auditPath, request);
  const state = { request_id: requestId, begin_at: begin, status: null,
    headers_at: null, first_byte_at: null, returned_model: null, stop_reason: null,
    raw_usage: {}, protocol: 'unparsed', unparsed_reason: 'fetch_not_returned' };
  let parser, terminal = false, providerError = false, ended = false;
  const record = (event, extra = {}) => append(responsePath, { event, at: now(), ...state, ...extra });
  function finish(transport) {
    if (ended) return;
    ended = true;
    if (transport === 'eof') parser?.finish();
    const parsed = state.protocol === 'anthropic_messages_sse';
    const failedHTTP = state.status !== null && state.status >= 400;
    record('response_end', { end_at: now(), elapsed_ms: performance.now() - started,
      // DSH returns on message_stop and can cancel its reader before EOF.
      // Protocol completion therefore does not require transport completion.
      transport, completed: failedHTTP || providerError ? false : parsed ? terminal : null,
      interrupted: !terminal && (['cancelled', 'stream_error', 'fetch_error'].includes(transport) ||
        (parsed && transport === 'eof' && !failedHTTP && !providerError)),
      parse_status: parsed ? (parser.stats.oversized_events || parser.stats.malformed_events ? 'partial' : 'parsed') : 'unparsed',
      ...(parser?.stats ?? {}), provider_error: providerError });
  }
  let response;
  try { response = await Reflect.apply(original, this, [input, init]); }
  catch (error) { finish('fetch_error'); throw error; }
  state.status = response.status;
  state.headers_at = now();
  const messages = url.pathname.endsWith('/messages');
  if (messages && response.headers.get('content-type')?.split(';')[0].trim() === 'text/event-stream') {
    state.protocol = 'anthropic_messages_sse';
    delete state.unparsed_reason;
    parser = messagesParser(value => {
      if (value.type === 'message_start') {
        if (typeof value.message?.model === 'string') state.returned_model = value.message.model;
        if (value.message?.usage) state.raw_usage.message_start = value.message.usage;
      } else if (value.type === 'message_delta') {
        if (typeof value.delta?.stop_reason === 'string') state.stop_reason = value.delta.stop_reason;
        if (value.usage) state.raw_usage.message_delta = value.usage;
      } else if (value.type === 'message_stop') terminal = true;
      else if (value.type === 'error') providerError = true;
      record('response_metadata', { source_event: value.type });
    });
  } else state.unparsed_reason = messages ? 'non_sse_content_type' : 'unsupported_route';
  record('response_start');
  if (!response.body) { finish('eof'); return response; }

  // One reader, driven only by the consumer. No clone/tee and no background
  // drain: cancellation and read errors retain the provider's normal semantics.
  const reader = response.body.getReader();
  const body = new ReadableStream({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (done) { finish('eof'); controller.close(); return; }
        if (value.byteLength) {
          if (state.first_byte_at === null) {
            state.first_byte_at = now();
            record('first_byte', { time_to_first_body_byte_ms: performance.now() - started });
          }
          parser?.write(value);
        }
        controller.enqueue(value);
      } catch (error) { finish('stream_error'); controller.error(error); }
    },
    cancel(reason) { finish('cancelled'); return reader.cancel(reason); },
  }, { highWaterMark: 0 });
  const observed = new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
  // Response's constructor otherwise loses these read-only fetch properties.
  for (const key of ['url', 'redirected', 'type']) Object.defineProperty(observed, key, { value: response[key] });
  return observed;
};
