# Three-harness DexJoCo pilot

## Goal and scope

Compare the two DeepSeek harnesses under the same task and provider conditions,
and compare both complete agent configurations with Cursor/Grok. Preserve the
official simulator, robot interface, layout split and scoring. This phase supports
the eleven standard DexJoCo tasks in native, single-agent, single-environment,
code-formal mode. Live-agent formal evaluation, subagents and other benchmarks
are outside this pilot.

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

## Current integration limits

DSH and Claude Code have passed live image/tool/shell/resume/max-effort checks
against the selected custom relay. All eleven simulator doctors pass. Goal-image
preparation and full task trials have separate machine-local progress records.

The account's Cursor catalog advertises a 500k window, xhigh and Fast, but the
tested CLI selection has not yet demonstrated all three settings together. An
observed fallback to High/Fast/default context is a failed configuration check.
The adapter implementation and direct native sandbox checks do not replace a
successful actual-agent configuration and boundary check. Until that succeeds,
the requested Cursor robotics trial must remain unstarted.

Keep source and sanitized evidence in Git; environments, account files, real API
keys, full transcripts, recordings and operator manifests stay outside it.
