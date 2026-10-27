# Three-harness DexJoCo pilot

## Goal and scope

Compare the two DeepSeek harnesses under the same task and provider conditions,
and compare both complete agent configurations with Cursor/Grok. Preserve the
official simulator, robot interface, layout split and scoring. This phase supports
the eleven standard DexJoCo tasks in native, single-agent, single-environment,
code-formal mode. Live-agent formal evaluation, subagents and other benchmarks
are outside this pilot. Configuration covers those eleven tasks; the end-to-end
integration acceptance reported below exercises Click Mouse only.

| Harness | Provider | Requested model | Effort |
| --- | --- | --- | --- |
| DSH | `custom` | `deepseek-v4.1-flash` | `max` |
| Claude Code | `custom`, same endpoint | `deepseek-v4.1-flash` | `max` |
| Cursor CLI | `cursor-account` | `grok-4.7[context=500k,reasoning_effort=xhigh,fast=true]` | `xhigh` |

The tested DeepSeek relay returned `deepseek-v4-1-flash-260910`; this is not
evidence of the official experiment's 1017 snapshot. Record both requested and
returned identifiers. Keep the model endpoint and private key outside Git.

## Required behavior

Each adapter must load the same task instructions and RGB-D robot tools, inspect
images, run code within the experiment boundary, preserve its own conversation
on resume, stop its child processes, and support rehearsal, submission, freezing
and official evaluation. Model credentials and simulator-private files must be
unreadable to generated code and native file tools. Do not enable unrestricted
Cursor commands or silently substitute a different model, effort or context tier.

Acceptance separates infrastructure from task performance: a working adapter
does not imply the model solves a task. An interrupted exploration without a
submitted controller has no held-out success rate.

## Prepare and run

Use the installed host environment and a pinned CLI binary. The same configuration
command applies to every standard task and all three harnesses:

```bash
CONFIG=$(runtime/envs/host/bin/python scripts/configure_auto_research.py \
  --benchmark dexjoco --task click_mouse --sim-gpu 0 \
  --agent-cli dsh --agent-provider custom --agent-effort max \
  --episodes 16 --exploration-envs 1 --subagents 0 \
  --observation-profile rgbd --demonstration-context terminal_state \
  --expert-demonstrations 0 --external-perception off --episode-reviews off \
  --formal-episodes 3 --formal-workers 1)

timeout --signal=TERM --kill-after=30s 7200 \
  bash scripts/start_auto_research_agent.sh --config "$CONFIG" \
  --agent-binary "$RAX_AGENT_BINARY" --agent-model deepseek-v4.1-flash \
  --agent-base-url "$RAX_MODEL_BASE_URL" --agent-key-file "$RAX_API_KEY_FILE" \
  --non-interactive
```

For Claude Code use `--agent-cli claude` with the same provider, model, effort
and generic launch flags. All auxiliary model aliases are mapped to the selected
DeepSeek model. The custom Messages transport keeps the real key on the host,
normalizes nested tool images and records request metadata, response usage and
timing without logging credentials or prompt contents.

The DSH transport also records the returned model identifier, response timing and
raw provider usage. Responses are correlated by request ID in a separate
`dsh-home/response-audit.jsonl`, preserving the existing request-count format.
Its first-byte measurement is client-observed body availability, not network-only
latency. A provider message-end marker establishes model completion even when
the native client then closes the stream. Missing end records after a process
kill remain unfinished; raw usage is not converted into billing estimates.

For Cursor use `--agent-cli cursor --agent-provider cursor-account
--agent-effort xhigh`. At launch use the requested model string above, a verified
project-local CLI binary and, optionally, `--agent-auth-home` pointing to the
authenticated XDG `cursor` directory. Do not pass DeepSeek endpoint/key flags.
Cursor uses private per-session config/data directories and its existing account
without copying auth into the workspace. The adapter uses native sandboxing and
allowlists, protects its control files, and denies subagent starts with a native
fail-closed hook. It has no container adapter.

The dashboard supports configuration, monitoring, stop and resume. Its existing
form has no endpoint/key/model/binary fields; custom-provider first launches use
the CLI commands above.

## Execution and stopping rules

1. Validate the model/image/tool and isolation contracts before robotics runs.
2. Validate all eleven simulators and prepare three successful terminal reference
   images per task, excluding every formal layout through the official tooling.
3. Run Click Mouse sequentially for each configuration, with at most two hours
   of active session time, including inference waits, tools, retries and formal.
4. Preserve the workspace and logs on stop. Before a permitted resume, subtract
   previous active attempts from the time allowance; do not reset its budget.
5. Evaluate only a frozen submitted controller, initially on three formal layouts.
   The 150-layout batches and independent repeats require a separate budget.

Keep preparation/integration probes separate from these trials. Do not select
only successful exploration sessions when reporting the comparison. Record
submission status, success rate when available, active time, model and tool time,
reported tokens/cache, compaction events and interruption reasons. Cursor result
usage is per completed turn; it is not a count of individual model requests.

The custom DeepSeek gateway's cache-accounting semantics are not yet verified.
Preserve its raw `input_tokens`, `cache_read_input_tokens`,
`cache_creation_input_tokens` and `output_tokens` fields. Native Claude and DSH
usage conversion assumes Anthropic input excludes cache; the observed gateway
values may instead include it. Do not use those converted totals as verified
billable tokens or draw cross-harness cost conclusions from them. Separate
completed responses from interrupted streams and report missing usage explicitly.

## Current integration evidence and limits

All eleven simulator doctors pass. DSH and Claude Code passed live image/tool,
shell, resume and max-effort checks against the selected relay. Their earlier
bounded task-solving trials did not submit a controller; those runs therefore
have no held-out success rate. Do not turn a missing submission into a scored 0%.

The 2026-10-27 check uses the separate procedure in
`native-integration-acceptance.md`. It explicitly asks the native agent to write
its own minimal hold controller and exercise registration, rehearsal, submission,
freezing and formal scoring. Its results establish workflow integration, not
robot-task-solving capability, and must not be pooled with benchmark results.

Claude Code 2.1.310 with requested `deepseek-v4.1-flash` / `max` completed this
check on Click Mouse: three complete and valid formal outcomes, zero task
successes, no formal execution or infrastructure errors. The relay returned
`deepseek-v4-1-flash-260910`. An initial rehearsal exposed system bubblewrap
0.6.1 lacking `--size`; the project launch environment now resolves the already
installed compatible binary. The agent resumed its native conversation and
submitted the same unchanged controller. Total active time, including the failed
attempt, was 612.931 seconds. No task-success requirement or scoring rule changed
inside the benchmark.

Cursor CLI 2026.10.22-27dcf5f requires official account/feature initialization in
its fresh private config directory. The adapter now uses ACP `initialize` and
`authenticate(cursor_login)` before checking model metadata.
Its model catalog also needs to be ready: native `--model` resolution can fall
back to the legacy list when the parameterized catalog misses its two-second
loading window. The failed run directly evidenced a legacy-only rejection; it
did not record the model fetch timing, so a specific two-second timeout on that
request is not established. Authentication alone does not establish catalog readiness.
The adapter creates an empty official ACP configuration session with the supported
parameterized-model-picker client capability, waits for the full catalog, selects
the requested model/context/effort/Fast options and verifies the returned options
and vendor-written persisted selection. It sends no `session/prompt` and does not
copy caches or override vendor feature gates. The native CLI then uses that saved
selection rather than reparsing `--model`; actual `system/init` must still match.
The metadata session is not the native experiment conversation.
It then verifies the effective workspace read boundary before starting the model.
The canary tests denied file contents rather than treating visible `/proc`
metadata as readable contents.

Cursor also discovers the project through the nearest `.git` marker. An invalid
ancestor marker on the test host hid RAX's workspace MCP configuration. Each newly
created Cursor agent workspace now has its own Git root; `.git` and `.cursor`
are protected against model writes. The adapter approves only its `auto_research`
MCP through the official CLI and preserves native tool permissions.

A real generation selected `Grok 4.7 500K Extra High Fast`, with persisted
`context=500k`, `reasoning_effort=xhigh`, `fast=true` and `maxMode=true`. The adapter
checks native initialization and stored selection and rejects a silent change.
This verifies the selected mode; it does not stress-test a 500k-token input.
Earlier catalog and selection failures remain historical evidence, not a claim
that this account or model can only support 256k/High/Fast. Available model metadata differed between observations; those records did not
isolate catalog loading, picker representation or server-side availability as
the cause of every earlier selection failure.

The adapter must not add `Shell(*)` to `permissions.allow`. Cursor treats an
allowlisted shell command as approved to run outside its sandbox, even when
sandbox mode is enabled. Removing that entry retains deterministic Allowlist
mode and sends supported commands through the native sandbox; no Auto-review
classifier or extra model is enabled. The actual native Shell canary, rather
than only `agent sandbox run`, must confirm these restrictions.

A startup failure before the native conversation ID now exits immediately. The
shared launcher no longer waits and retries a new session with a continuation
prompt when `latest_session` returns no conversation. Existing valid-session
resume behavior is unchanged.

Cursor completed the same Click Mouse workflow only after an operator-directed
recovery. The original frozen controller had four healthy rehearsals, but the
native CLI reconnected/retried without completing the long `rehearse` tool reply
while RAX suspended the whole client with SIGSTOP. The exact first triggering
timer is unproven. A single bounded metadata probe did not resolve that cause.
A subsequent instruction resumed the same native conversation, required `status`
confirmation of the existing rehearsal qualification and asked the agent to
submit that unchanged bundle. The agent itself called `status` and `submit`, then
ended its turn. The original driver ran all three formal layouts: complete and
valid native outcomes, zero successes, zero execution/infrastructure errors.

Total active time including prior failed attempts and the diagnostic probe was
1672.123 seconds. The independent audit passed 29 checks; one agent-registered
controller and one system overall checkpoint are distinct stored entries.
This is `passed_with_directed_recovery`, not an unattended adapter acceptance.
The long synchronous MCP response problem remains unresolved. No asynchronous
protocol, vendor timeout patch, operator submission or scoring change was made.
Do not present these passive-controller 0/3 outcomes as model capability scores.
See `integration-repair-results-20261027.md` for the bounded result record.

Keep source and sanitized evidence in Git; environments, account files, real API
keys, full transcripts, recordings and operator manifests stay outside it.
