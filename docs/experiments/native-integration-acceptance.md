# Native harness integration acceptance

Use this check to establish that a native coding-agent harness can complete the
RAX code-formal lifecycle. Robot task success is not required. This is a separate
integration experiment, not a benchmark capability score.

The ordinary task prompt asks the agent to achieve manual success before writing
an autonomous controller. Failure to solve that first stage can prevent a useful
adapter check from reaching registration or formal evaluation. The companion
`integration-acceptance-prompt.txt` explicitly replaces that performance
prerequisite for this check only; the normal benchmark instructions are unchanged.

## Run

Prepare a fresh native-agent configuration using the task and model settings from
`three-harness-pilot.md`. Keep one agent, one environment, 16 shared exploration
and rehearsal episodes, three formal episodes, and the original simulator,
observation, isolation and scoring settings. A disposable `--prepare-only` check
must use a different session directory from the actual run.

The launcher's host `PATH` must resolve a bubblewrap binary supporting `--size`,
which the existing controller sandbox uses to bound its scratch filesystem.
An older installed `bwrap` can launch the native agent but fail only when the
first controller rehearsal starts. Use a verified project-local compatible binary;
do not remove the sandbox option or change system security policy to get past it.
Both normal code-formal startup and `--prepare-only` check this capability before
deployment or model execution.

Pass the companion prompt through the existing `--exec-prompt` option when
launching the configured native harness. For example, after setting the model,
endpoint and private key-file variables for Claude Code:

```bash
RAX_INTEGRATION_PROMPT="$(cat docs/experiments/integration-acceptance-prompt.txt)"
timeout --signal=TERM --kill-after=30s 7200 \
  runtime/envs/host/bin/python -m harness.codex_cli.auto_research \
  --config "$RAX_CONFIG" --non-interactive \
  --agent-binary "$RAX_AGENT_BINARY" --agent-model "$RAX_AGENT_MODEL" \
  --agent-base-url "$RAX_MODEL_BASE_URL" --agent-key-file "$RAX_API_KEY_FILE" \
  --exec-prompt "$RAX_INTEGRATION_PROMPT"
```

For Cursor, use its configured account and native CLI; omit the endpoint/key
flags and use `--agent-auth-home` when needed. Verify the requested effective
model, effort, context and Fast settings and the actual native isolation boundary
before a robotics run. An advertised catalog option or adapter unit test does not
establish that those settings or boundaries took effect.

For the native Cursor CLI, leave shell commands out of `permissions.allow` while
keeping Allowlist mode and sandboxing enabled. Explicit shell allowlisting means
permission to execute outside Cursor's sandbox. Check actual model-issued Shell
calls as well as `agent sandbox run`; the latter alone cannot detect a bypass
caused by tool approval policy. Use nonsecret fixtures to verify permitted public
reads, denied private contents, readonly control files, and blocked network/GPU
access. Do not enable `--force`, unrestricted mode or an extra Auto-review model
to bypass a failing check. See the official
[Run Modes documentation](https://cursor.com/docs/agent/security/run-modes).


The two-hour active-run cap includes model waits, tools, actual retry attempts
and formal work. Offline code inspection, editing and decision waits are excluded
and recorded separately; active time is not an end-to-end latency comparison. A repeated
attempt uses only the remaining cumulative budget. The experiment supervisor
must retain process and budget records and clean up only its own descendants.
Do not restart automatically or extend the budget after a failure.

## What the agent must do

The native agent reads the actual instructions, observes real robot images,
executes a short legal action, and writes its own small controller. The controller
uses the supplied `ctx`, observes the environment and performs legal hold actions
until the native episode ends. It then returns without another robot call.

An immediate return is insufficient: it may produce a false task-completion
value while the native outcome still says `complete=false` and
`valid_for_success_rate=false`. For DexJoCo, use the documented `dexjoco_hold`
tool, at most 150 steps per call, respect the public episode limits and stop on
`episode_ended`. The integration prompt supplies this procedure, not a solved
task controller or a fabricated score.

The agent registers the controller directory, rehearses the exact frozen bundle,
checks qualification, submits it, and ends its turn. The existing launcher then
freezes the exploration session and runs the three formal layouts without model
intervention. Do not add operator auto-submission or a fallback controller.

## Acceptance

Require all of the following, independently of whether the task succeeds:

- The requested native model configuration and isolation checks pass.
- Native images, shell/file tools and robot calls are evidenced by the actual run.
- A frozen bundle is registered, rehearsed and submitted by the tested agent.
- The qualifying rehearsal and every formal run use that exact bundle and hash.
- All three formal episodes finish, with complete, valid native scores and no
  execution, recording or infrastructure errors.
- The launcher, supervisor and their simulator children exit and release resources.

A healthy, fully scored `0/3` satisfies this integration test. No submission,
missing scores, an early-return incomplete native episode, or a process exit code
of zero alone does not. Keep integration outcomes separate from benchmark trials
and preserve the exact prompt, source commit, CLI version, model settings, raw
usage semantics and result evidence.

## Recovery and incomplete adapter acceptance

Record any operator recovery instruction separately from the initial prompt. A
completed formal batch after such an instruction establishes that the submitted
program and formal driver work, but it does not repair the interrupted tool
round trip or establish unattended operation. Preserve prior failures and the
cumulative active budget; the agent must still check the official qualification
and submit the exact frozen controller itself.

In the 2026-10-27 Cursor run, whole-client SIGSTOP during a long rehearsal caused
observed reconnect/retry behavior, and the native agent did not receive completed
rehearsal replies. Directed same-session `status` then `submit` recovered the
workflow. Mark `unattended_full_adapter_passed=false` and
`long_mcp_transport_fixed=false` until a normal long rehearsal returns to the
agent and it proceeds without that intervention. Any asynchronous replacement
must independently validate the experiment's tool/isolation boundary and disclose
the changed protocol rather than claim equivalence with whole-process freezing.
