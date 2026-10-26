"""Native DSH provider configuration; credentials stay outside experiment records."""
from pathlib import Path
from urllib.parse import urlsplit

from harness.claude_cli.login import private_text

PROVIDERS = ('openrouter', 'deepseek-api', 'deepseek-account', 'custom')
MODEL = 'deepseek/deepseek-v4.1-flash-1017'


class Credential:
    def __init__(self, provider, *, key_file=None, auth_home=None, base_url=None):
        if provider not in PROVIDERS:
            raise ValueError(f'Unknown DSH provider {provider!r}')
        self.provider = provider
        self.key_file = Path(key_file).resolve() if key_file and provider != 'deepseek-account' else None
        self.auth_home = Path(auth_home).resolve() if auth_home else None
        self.base_url = base_url
        if provider == 'deepseek-account':
            if not self.auth_home:
                raise ValueError('deepseek-account requires a dedicated --agent-auth-home')
            self.credentials_file = self.auth_home/'.credentials.yaml'
            private_text(self.credentials_file)  # Check owner/mode without publishing contents.
        else:
            if not self.key_file or not private_text(self.key_file).strip():
                raise ValueError(f'{provider} requires a nonempty private key file')
        if provider == 'custom':
            parsed = urlsplit(base_url or '')
            if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError('custom needs an HTTP(S) base URL without embedded credentials')

    def record(self):
        return {'provider': self.provider, 'key_file': str(self.key_file) if self.key_file else None,
                'auth_home': str(self.auth_home) if self.auth_home else None, 'base_url': self.base_url}

    def environment(self):
        return {} if self.provider == 'deepseek-account' else {'RAX_MODEL_KEY': private_text(self.key_file).strip()}

    def patches(self, model, effort):
        if self.provider == 'custom':
            return 'rax-private', [{'insert': [{'id': 'rax-private',
                'name': str(Path(__file__).with_name('private_provider.mjs')),
                'config': {'baseURL': self.base_url, 'thinking': 'enabled', 'reasoningEffort': 'max',
                    'models': [{'id': model, 'name': model, 'inputModalities': ['text', 'image']}]}}]}]
        if self.provider == 'openrouter':
            route = 'rax-model'
            provider = {'api': 'openai-completions',
                'baseURL': 'https://openrouter.ai/api/v1',
                'apiKeyEnv': 'RAX_MODEL_KEY', 'reasoning': effort,
                'models': [{'id': model, 'name': model, 'contextWindow': 1048576,
                            'maxTokens': 393216, 'input': ['text', 'image'],
                            'reasoningEfforts': {'max': 'max', 'xhigh': 'max'}}]}
            return route, [{'id': 'llm-pi-ai', 'config': {'providers': {route: provider}}}]
        route = 'deepseek-account' if self.provider == 'deepseek-account' else 'deepseek-official'
        # Filled from the native adapter's schema, not an emulated account token.
        rows = [{'id': 'llm-deepseek-account' if self.provider == 'deepseek-account' else 'llm-deepseek',
                 'config': {'thinking': 'enabled', 'reasoningEffort': 'max',
                            **({'apiKeyEnv': 'RAX_MODEL_KEY'} if self.provider == 'deepseek-api' else {})}}]
        if self.provider == 'deepseek-account':
            rows += [{'id': key, 'disabled': False} for key in ('credentials', 'authorization', 'deepseek-account')]
            rows += [{'id': 'credentials', 'config': {'path': str(self.credentials_file)}}]
        return route, rows
