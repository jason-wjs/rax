# Linux pilot: DSH / DeepSeek and Cursor / Grok

This pilot uses the personal fork as `origin`, the original RAX repository as
`upstream`, and `experiment/cursor-dsh-smoke` based on `dexjoco/native`. Keep
runtime environments, credentials, model transcripts and recordings outside Git.
The official DexJoCo checkout stays at `dependencies.lock` without modifications.

## Project layout

The machine-local lab directory contains `rax/` (source), `runtime/` (environments,
simulator and sessions), `logs/`, `experiments/` (private metadata) and `env.sh`.
Source `../env.sh` before running the commands below from the repository root.
Only small reusable configuration and sanitized evidence belong in this fork.

## Install and verify

```bash
bash scripts/bootstrap_sources.sh
bash scripts/setup_host_env.sh
PIP_CONSTRAINT="$PWD/experiments/dexjoco/constraints.txt" bash scripts/setup_dexjoco_env.sh
runtime/envs/host/bin/python scripts/install_dsh.py \
  --prefix "$RAX_RUNTIME_ROOT/agent-backends/deepseek-harness/tools/dsh-0.2.1-rc.2" \
  --cache "$RAX_RUNTIME_ROOT/cache"

runtime/envs/host/bin/python scripts/dexjoco_doctor.py --gpu 0 \
  --task click_mouse --task bimanual_assembly
```

The additional constraint retains official NumPy 1.26.4 and selects compatible
JAX/JAXlib 0.6.2 and OpenCV 4.11.0.86. This avoids dependency resolver backtracking;
it does not change simulator source, dynamics or task definitions.

Verified on the RTX 4070 Laptop 8 GB GPU:

- Host and simulator environments pass `pip check`.
- 47 backend, benchmark-registry and workspace-validation tests pass.
- 13 DSH adapter/image-layout regression tests pass.
- The official doctor passes single-arm and bimanual scenes: 640x640 RGB-D,
  calibration, motion chunks, recordings and three-layout determinism checks.
- Sanitized evidence is in `experiments/dexjoco/4070-simulator-smoke.json`.

Start with one exploration environment and one formal worker on this GPU.

## DSH engineering session

Use an operator-supplied private key file (mode 0600) and Messages-compatible
endpoint. Set `RAX_API_KEY_FILE` and `RAX_MODEL_BASE_URL` outside Git. The tested
relay advertises `deepseek-v4.1-flash`; its returned identifier was
`deepseek-v4-1-flash-260910`, which must not be described as the paper's 1017 snapshot.

```bash
CONFIG=$(runtime/envs/host/bin/python scripts/configure_auto_research.py \
  --benchmark dexjoco --task click_mouse --sim-gpu 0 \
  --agent-cli dsh --agent-provider custom --agent-effort max \
  --episodes 4 --exploration-envs 1 --subagents 0 \
  --observation-profile rgbd --external-perception off --episode-reviews off \
  --demonstration-context none --expert-demonstrations 0 \
  --formal-episodes 3 --formal-workers 1)

timeout --signal=TERM --kill-after=30s 30m \
  bash scripts/start_auto_research_agent.sh --config "$CONFIG" \
  --agent-binary "$RAX_RUNTIME_ROOT/agent-backends/deepseek-harness/tools/dsh-0.2.1-rc.2/node_modules/.bin/dsh" \
  --agent-model deepseek-v4.1-flash --agent-base-url "$RAX_MODEL_BASE_URL" \
  --agent-key-file "$RAX_API_KEY_FILE" --non-interactive
```

This small protocol has no goal-reference images, expert trajectories, external VLM
or episode reviewer. It checks integration and is not a formal benchmark comparison.
The bounded run executed real hand/arm actions but reached the 30-minute external
limit before confirming task success or submitting a frozen controller. No held-out
success rate is available. Its workspace, transcript and recordings remain resumable.

## Required custom-provider fixes

Custom model identifiers must be registered with text and image capabilities;
otherwise DSH's advisory catalog defaults unknown identifiers to text-only.

The tested Messages relay also mishandles images nested in tool_result content.
For the same 640x640 PNG, it reported 309 input tokens in a normal user image block,
231793 when nested in a tool result, and 611 when the tool result retained text and
the image moved to the containing user message. These are endpoint-reported counters,
not independent measurements of model tokenization or billing.

The custom transport now lifts tool images while preserving tool_use_id, error
status, existing text and image order. Evidence is recorded in
`experiments/dexjoco/relay-image-compatibility.json`. Earlier diagnostic sessions are
kept separately and must not be mixed with the corrected pilot.

## Cursor status and next decisions

The authenticated CLI lists `grok-4.7-xhigh`. A restricted MCP probe completed an
image/tool roundtrip and read its random token and colors correctly; this was not a
robotics or vision benchmark. A full Cursor RAX adapter has not yet been implemented.

Cursor native sandbox preflight fails with namespace EPERM on the current host,
although ordinary unshare and system bubblewrap work. The relevant AppArmor profile
comments out userns; the installed 3.0.4 parser rejects that rule in a parse-only
candidate test. No system security settings were changed. The MCP-only probe used
explicit allowlists with shell, file and web tools denied. Equivalent shell isolation
must be validated before a full Cursor experiment.

Before formal evaluation, agree on the task list, goal-reference images, independent
session repeats, exploration and wall-clock budgets, quota accounting and treatment
of infrastructure interruptions. Record source, simulator, CLI, model and provider
identifiers with every run. Keep time-limited engineering runs separate from scores.
