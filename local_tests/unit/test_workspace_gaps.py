"""Regression tests for gaps found in reviews of the generated workspaces."""

import pytest

from services.controller.frontend import DEFINITIONS, for_cli
from services.mcp_contract import Contract
from services.workspace import describe
from services.workspace.contract_files import manifest
from bench import STATUS


@pytest.mark.parametrize('cli', ['claude', 'dsh', 'kimi', 'cursor'])
def test_native_tool_descriptions_have_no_codex_wrapper(cli):
    assert any('functions.exec' in t['description'] for t in DEFINITIONS)
    assert not any('functions.exec' in t['description'] for t in for_cli(DEFINITIONS, cli))
    assert for_cli(DEFINITIONS, 'codex') == DEFINITIONS


def test_contract_file_names_the_role_and_every_tool():
    copy = manifest(Contract('isolated'), {'formal-episode'})
    assert copy['role'] == 'live-copy' and 'status' in copy['tools'] and 'submit' not in copy['tools']
    assert manifest(Contract())['role'] == 'exploration'
    assert manifest(Contract('isolated'))['role'] == 'formal-worker'


def test_session_file_explains_lookalike_tools():
    text = describe(Contract())
    assert '`status` reports budgets, bundles' in text and f'`{STATUS}` reports the live' in text
