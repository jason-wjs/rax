# Native integration results — 2026-10-27

These runs check the RAX lifecycle, not robot-task-solving capability. Both real
native agents were instructed to write a legal hold controller until the native
episode ended. The normal benchmark task prompt was not changed.

| Configuration | Workflow result | Formal outcomes | Active time |
| --- | --- | --- | --- |
| Claude Code 2.1.310 + DeepSeek V4.1 Flash / max | Passed after environment repair and same-session resume | 3 complete/valid, 0 successes, 0 errors | 612.931 s |
| Cursor CLI 2026.10.22-27dcf5f + Grok 4.7 / 500k / xhigh / Fast | Completed with operator-directed recovery; long MCP response unresolved | 3 complete/valid, 0 successes, 0 errors | 1672.123 s |

Both ran DexJoCo Click Mouse on Linux / RTX 4070, single agent/environment,
with 16 shared exploration/rehearsal episodes and three formal layouts. Each
formal episode ended naturally after 1170 steps, exit code 0, with native
`complete=true` and `valid_for_success_rate=true`. The layouts were collection 1,
IDs 0/1/2 (native seed 0, episode indices 0/1/2), not the full 150-layout schedule.
These passive-controller 0/3 results must not be compared with task-solving runs
such as the earlier Codex Hammer Nail 3/3.

## Evidence and changes

Claude's first actual rehearsal exposed a host bubblewrap 0.6.1 binary lacking
`--size`. The project environment selected an already installed compatible binary;
the same native conversation and unchanged controller then completed rehearsal,
submission and the formal batch. All 35 audited requests specified max effort.
The relay returned `deepseek-v4-1-flash-260910`; this does not establish identity
with the official RAX 1017 snapshot or independently verify the relay's provider.

Cursor's verified effective selection was `Grok 4.7 500K Extra High Fast`, including
maxMode. Official ACP parameter selection initializes private state before native
execution. An owned Git root makes the workspace MCP visible; only `auto_research`
is approved. Removing the `Shell(*)` allowlist entry restores native sandboxing.
Actual model-issued file/shell calls and image/MCP returns were checked. This
verifies the mode selection, not 500k-input capacity under load.

Cursor had four healthy controller rehearsals, but the native tool replies failed
under whole-client SIGSTOP and the client reconnected/retried. A bounded probe did
not establish which of the native timeout/heartbeat mechanisms triggered first.
The same native conversation was subsequently instructed to check `status` and
submit the already-qualified original bundle. The native agent performed both
calls; the operator neither wrote nor submitted the controller. The original RAX
driver then ran the formal batch. No freeze-protocol or scoring change was made.

Consequently Cursor remains `unattended_full_adapter_passed=false` and
`long_mcp_transport_fixed=false`. An asynchronous long-tool protocol is a proposed
follow-up requiring separate isolation validation and explicit protocol labeling.
It is not implemented by these changes.

Independent audits passed 15 Claude checks and 29 Cursor checks, including exact
bundle/execution-copy hashes, native formal scores, retained repair history and
owned-process/GPU cleanup. Regression batches passed 113 adapter/lifecycle/config
checks and 3 transport-metadata checks. Active time includes actual attempts and
formal work; offline investigation and decision waits are excluded.

## Frozen controller identity

| Profile | Bundle | controller.py SHA-256 | Final run source |
| --- | --- | --- | --- |
| Claude | `43c3024d707b4dd3a226913fe622c25b` | `50683787ec59d2ea31af73032a83bfb372a4ef09b3c3a345861d1106e5250fed` | `01b1d204abf22af42560002924f9250562ee93ef` |
| Cursor | `8770b56e7ba044a5921d5f2c128577e9` | `21239f657475c82e52dd5ce58f13779aae21cfb0f088470c966ca3d2d04f401d` | `b32f8959392f6dd93c627896ba51f47d8a903b55` |

See [the acceptance procedure](native-integration-acceptance.md) and
[the pilot configuration](three-harness-pilot.md). Private credentials, account
homes, full transcripts and model reasoning are excluded from source and exports.

## Protocol decision — 2026-10-27

The user selected the original synchronous whole-agent pause protocol. Cursor
long-rehearsal support is temporarily **blocked**; no asynchronous protocol is
authorized or implemented. No additional Cursor long-rehearsal trials or
automatic retries are scheduled in this phase. Preserve the completed formal
results as workflow completion with operator-directed recovery, not unattended
adapter acceptance. Reassessment requires new compatible evidence and a
subsequent user instruction.
