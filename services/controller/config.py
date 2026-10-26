"""Operator-owned configuration. No values are accepted from controller code."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import re

from services.benchmarks import DEFAULT as DEFAULT_BENCHMARK
from services.demonstration_io import DEMONSTRATION_CONTEXTS

FORMAL_EPISODES = 50  # Default held-out formal episodes of a new session.
# Reasoning efforts both agent CLIs (Codex model_reasoning_effort, Claude Code --effort)
# and OpenRouter accept; xhigh is the established default.
AGENT_EFFORTS = ('high', 'xhigh', 'max')
# What the mounted expert episodes hold: 'full' (commanded references, measured motion, object
# states and camera) or 'images' (front-camera video, previews and depth only).
EXPERT_CONTENTS = ('full', 'images')
# What TASK.md gives of the task's rules: 'public' (the benchmark's public task definition), or
# 'code-summary' (plus rules summarized from the task code), where a benchmark offers it.
TASK_RULES = ('public', 'code-summary')
# Exploration episodes beyond a collection's layouts cycle through it again.
MAX_EXPLORATION_EPISODES = 1000
# The largest formal set a session may evaluate.
MAX_FORMAL_EPISODES = 150
# mode, student_*, teacher_source_session: configurations saved while other experiment modes
# existed. agent_formal_max_model_calls: live copies no longer have a model-call cap.
# controller_gpu, training_gpu: agents and bundles no longer get a GPU; only the simulator GPU
# remains. image, workspace_mb: agents and bundles run natively, without a container image or a
# capped workspace filesystem.
# agent_account: claude-login sessions use the one experiment account; there is no account pool.
RETIRED = ('mode', 'student_model', 'student_reasoning_effort', 'student_max_model_calls', 'teacher_source_session',
           'agent_formal_max_model_calls', 'controller_gpu', 'training_gpu', 'image', 'workspace_mb',
           'agent_account')


def with_task_timeouts(config):
    from services.benchmarks import for_config
    seconds = for_config(config).task_timeouts(config.task)['episode_seconds']
    return replace(config, development=replace(config.development, wall_seconds=seconds),
                   formal=replace(config.formal, wall_seconds=seconds))


def retire_fields(value):
    """A saved configuration without the retired fields; a removed mode is a clear error."""
    value = dict(value)
    mode = value.get('mode', 'auto-research')
    if mode != 'auto-research':
        raise ValueError(f'Experiment mode {mode!r} was removed; only auto-research remains')
    for key in RETIRED:
        value.pop(key, None)
    # Existing sessions exposed the external perception tool before this setting existed.
    value.setdefault('external_perception', True)
    return value


@dataclass(frozen=True)
class Limits:
    wall_seconds: int | None  # None: no wall-clock limit (the episode ends at its step limit).
    memory_mb: int
    cpus: float
    pids: int
    scratch_mb: int
    artifact_bytes: int

    def __post_init__(self):
        for key, value in vars(self).items():
            if key == "cpus":
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                    raise ValueError("cpus must be positive and finite")
            elif key == "wall_seconds" and value is None:
                continue
            elif type(value) is not int or value < 1:
                raise ValueError(f"Invalid limit: {key}")


@dataclass(frozen=True)
class Configuration:
    root: Path
    task: str
    exploration_seeds: tuple[int, ...]
    formal_seed: int
    eval_seed: int
    sim_gpu: str
    development: Limits
    formal: Limits
    training: Limits
    demonstration_context: str = 'none'  # Preserve older configurations; operator UI/CLI default to final image.
    # Complete expert episodes mounted read-only at runtime/experts (benchmarks that have them).
    expert_demonstrations: int = 0
    expert_content: str = 'full'  # EXPERT_CONTENTS
    formal_episodes: int = 1  # Legacy configurations remain single-episode.
    formal_eval_seed: int | None = None
    formal_workers: int = 1  # Concurrent formal episodes; legacy configurations stay sequential.
    # Agent formal only: the GPUs its copies' simulators take in turn (episode i on entry i mod n);
    # empty, every simulator runs on sim_gpu.
    formal_sim_gpus: tuple[str, ...] = ()
    observation_profile: str = 'rgbd'
    exploration_envs: int = 1
    subagents: int = 0  # Concurrent exploration subagents; 0 = one per environment when envs > 1, else none
    task_rules: str = 'public'  # TASK_RULES
    agent_cli: str = 'codex'  # 'codex' or 'claude' (harness/claude_cli)
    agent_provider: str = 'openrouter'  # Provider choices depend on the native agent backend.
    external_perception: bool = True  # Optional external Gemini tool; native model vision works without it.
    agent_effort: str = 'xhigh'  # Native reasoning effort, validated for the selected adapter.
    # The session's one evaluation mode. Off (code mode): the agent submits one rehearsed
    # bundle, run alone on the held-out layouts. On (live-copy mode, services/agent_formal):
    # exploration has no bundle tools; once its episodes are spent, each held-out episode is
    # played live by a copy of the agent that carries only the transfer folders and, unless
    # formal_context is 'clear', its conversation.
    agent_formal: bool = False
    write_handoff: bool = False  # Install the transfer-handoff skill (source trials).
    formal_context: str = 'compact'  # resume | compact | clear
    transfer_parts: tuple[str, ...] = ('memory', 'code', 'handoff')
    transfer_mb: int = 100  # Cap on the transferred folders (not the transcript).
    # Adaptation (services/adaptation.py): continue a finished source trial's agent in this
    # trial's environment (its task and eval_seed), carrying transfer_parts and, unless
    # adaptation_context is 'clear', its conversation.
    adaptation_source: str | None = None
    adaptation_context: str = 'compact'  # resume | compact | clear
    # Simulator benchmark (services/benchmarks.py); configurations without it use its DEFAULT.
    benchmark: str = DEFAULT_BENCHMARK

    @property
    def formal_collection(self):
        return self.eval_seed if self.formal_eval_seed is None else self.formal_eval_seed

    @classmethod
    def from_dict(cls, value):
        value = retire_fields(value)
        value.pop("training_seconds", None)  # Retired, unenforced training allowance.
        value["root"] = Path(value["root"]).resolve()
        value["exploration_seeds"] = tuple(value["exploration_seeds"])
        for key in ("transfer_parts", "formal_sim_gpus"):
            if key in value:
                value[key] = tuple(value[key])
        for key in ("development", "formal", "training"):
            # Legacy operator files may include superseded phase API caps.
            # Gemini admission now uses only the shared session dollar ledger.
            limits = dict(value[key])
            limits.pop("gemini_calls", None)
            limits.pop("gemini_tokens", None)
            value[key] = Limits(**limits)
        return cls(**value)

    def __post_init__(self):
        from services.mcp_contract import Contract
        Contract.from_config(self)
        from services.benchmarks import for_config
        bench = for_config(self)
        if type(self.expert_demonstrations) is not int or self.expert_demonstrations < 0:
            raise ValueError('expert_demonstrations must be a nonnegative integer')
        if self.expert_demonstrations and 'expert_demonstrations' in bench.fixed:
            raise ValueError(bench.fixed['expert_demonstrations'][1])
        if self.expert_content not in EXPERT_CONTENTS:
            raise ValueError(f'expert_content must be one of {EXPERT_CONTENTS}')
        if self.expert_content != 'full' and not self.expert_demonstrations:
            raise ValueError('expert_content applies only with expert_demonstrations')
        bench.validate_config(self)
        if self.task_rules not in TASK_RULES:
            raise ValueError(f'task_rules must be one of {TASK_RULES}')
        from harness.backends import NAMES, PROVIDERS, provider_choices
        if self.agent_cli not in NAMES:
            raise ValueError(f'agent_cli must be one of {NAMES}')
        if type(self.external_perception) is not bool:
            raise ValueError('external_perception must be boolean')
        providers = provider_choices(self.agent_cli)
        if self.agent_provider not in providers:
            raise ValueError(f'agent_provider must be one of {providers}')
        if self.agent_cli == 'dsh' and (self.agent_formal or self.adaptation_source or self.subagents or self.exploration_envs != 1):
            raise ValueError('DSH currently supports code formal, one exploration environment, no adaptation or subagents')
        if self.agent_cli in PROVIDERS and (self.agent_formal or self.adaptation_source or self.subagents or self.exploration_envs != 1):
            raise ValueError(f'{self.agent_cli} currently supports code formal, one exploration environment, no adaptation or subagents')
        if self.agent_effort not in AGENT_EFFORTS:
            raise ValueError(f'agent_effort must be one of {AGENT_EFFORTS}')
        if self.agent_effort == 'high' and self.agent_cli not in ('codex', 'claude'):
            raise ValueError('high effort is currently supported only by Codex and Claude')
        if self.demonstration_context not in DEMONSTRATION_CONTEXTS:
            raise ValueError('Invalid demonstration_context')
        from services.transfer import CONTEXTS, validate_parts
        if type(self.write_handoff) is not bool:
            raise ValueError('write_handoff must be true or false')
        if type(self.agent_formal) is not bool:
            raise ValueError('agent_formal must be true or false')
        if self.formal_context not in CONTEXTS:
            raise ValueError(f'formal_context must be one of {CONTEXTS}')
        validate_parts(self.transfer_parts)
        if type(self.transfer_mb) is not int or not 1 <= self.transfer_mb <= 1024:
            raise ValueError('transfer_mb must be 1..1024')
        if self.adaptation_context not in CONTEXTS:
            raise ValueError(f'adaptation_context must be one of {CONTEXTS}')
        if self.adaptation_source is not None:
            from services.storage_root import artifact_root
            if (not isinstance(self.adaptation_source, str) or not Path(self.adaptation_source).is_absolute()
                    or not Path(self.adaptation_source).is_relative_to(artifact_root())):
                raise ValueError(f'Adaptation source must be an absolute trial path under {artifact_root()}')
        from services.storage_root import artifact_root, ENVIRONMENT
        if not self.root.is_relative_to(artifact_root()):
            raise ValueError(f"Controller artifacts must be under {artifact_root()} (set {ENVIRONMENT} to change it)")
        if not re.fullmatch(r"[A-Za-z0-9_]+", self.task):
            raise ValueError("Invalid task")
        if not self.exploration_seeds or len(self.exploration_seeds) > MAX_EXPLORATION_EPISODES:
            raise ValueError(f"Provide 1..{MAX_EXPLORATION_EPISODES} exploration episodes")
        if any(type(s) is not int or s < 0 for s in (*self.exploration_seeds, self.formal_seed, self.eval_seed)):
            raise ValueError("Seeds must be nonnegative integers")
        if type(self.formal_episodes) is not int or not 1 <= self.formal_episodes <= MAX_FORMAL_EPISODES:
            raise ValueError(f'formal_episodes must be 1..{MAX_FORMAL_EPISODES}')
        if type(self.formal_workers) is not int or not 1 <= self.formal_workers <= 16:
            raise ValueError('formal_workers must be 1..16')
        if self.formal_eval_seed is not None and (type(self.formal_eval_seed) is not int or self.formal_eval_seed < 0):
            raise ValueError('formal_eval_seed must be a nonnegative integer')
        if self.formal_collection == self.eval_seed and set(range(self.formal_seed, self.formal_seed+self.formal_episodes)).intersection(self.exploration_seeds):
            raise ValueError("Formal layout must be held out")
        if (type(self.formal_sim_gpus) is not tuple or len(set(self.formal_sim_gpus)) != len(self.formal_sim_gpus)
                or not all(isinstance(gpu, str) for gpu in self.formal_sim_gpus)):
            raise ValueError('formal_sim_gpus must be distinct GPU UUIDs')
        if self.formal_sim_gpus and not self.agent_formal:
            raise ValueError('formal_sim_gpus applies to agent formal only')
        if not self.sim_gpu:
            raise ValueError("A simulator GPU is required")
        for gpu in (self.sim_gpu, *self.formal_sim_gpus):
            if not isinstance(gpu, str) or not re.fullmatch(r"GPU-[a-fA-F0-9-]+", gpu):
                raise ValueError("Use an explicit GPU UUID, not an ordinal or 'all'")
