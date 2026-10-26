"""Agent adapters. Import an adapter only when its session needs it."""
from importlib import import_module

NAMES = ('codex', 'claude', 'dsh', 'kimi', 'cursor')
PROVIDERS = {
    'kimi': ('openrouter', 'moonshot-api', 'kimi-account', 'custom'),
    'cursor': ('cursor-account',),
}
MODULES = {
    'codex': 'harness.codex_cli.auto_research',
    'claude': 'harness.claude_cli.auto_research',
    'dsh': 'harness.deepseek_cli.auto_research',
    'kimi': 'harness.kimi_cli.auto_research',
    'cursor': 'harness.cursor_cli.auto_research',
}


def provider_choices(name):
    return PROVIDERS.get(name, ('openrouter', 'deepseek-api', 'deepseek-account', 'custom') if name == 'dsh'
                         else ('openrouter', 'claude-login', 'anthropic', 'custom'))


def get(name):
    try:
        module = MODULES[name]
    except KeyError:
        raise ValueError(f'Unknown agent backend {name!r}; choose from {NAMES}') from None
    from harness.cli_runtime import CONTAINER
    if CONTAINER and name == 'cursor':
        raise ValueError('Cursor currently supports native execution only')
    if CONTAINER and name not in ('codex', 'claude'):
        from harness.container_cli.adapter import Adapter
        return Adapter(name, import_module(module))
    return import_module(module)
