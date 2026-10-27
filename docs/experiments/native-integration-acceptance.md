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

The two-hour cap includes model waits, tools, repairs and formal work. A repeated
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
