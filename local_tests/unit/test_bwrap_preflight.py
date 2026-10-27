"""Reject an incompatible bundle sandbox before deployment or model execution."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from harness.codex_cli import auto_research as launcher
from services.controller import sandbox


OLD_HELP = 'Usage: bwrap [OPTIONS...] COMMAND\n    --tmpfs DEST Create a tmpfs mount\n'
NEW_HELP = OLD_HELP + '    --size BYTES Set the size of the next tmpfs\n'


class BwrapPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = self.root / 'python-environment'
        (environment / 'bin').mkdir(parents=True)
        (environment / 'bin/python').touch()
        self.patches = self.enterContext(ExitStack())
        self.patches.enter_context(patch.object(sandbox, 'python_environment', return_value=environment))
        self.patches.enter_context(patch.object(sandbox.shutil, 'which', return_value='/test/bin/bwrap'))
        self.help = self.patches.enter_context(patch.object(sandbox.subprocess, 'run'))

    def help_output(self, value):
        self.help.return_value = subprocess.CompletedProcess(['/test/bin/bwrap', '--help'], 0,
                                                            stdout=value, stderr='')

    def test_old_bwrap_fails_with_required_capability_and_selected_path(self):
        self.help_output(OLD_HELP)
        with self.assertRaisesRegex(RuntimeError, r'/test/bin/bwrap.*does not support --size'):
            sandbox.BundleSandbox(None).preflight()

    def test_supported_bwrap_passes_without_running_a_sandbox(self):
        self.help_output(NEW_HELP)
        sandbox.BundleSandbox(None).preflight()
        self.help.assert_called_once_with(['/test/bin/bwrap', '--help'], capture_output=True,
                                          text=True, check=True, timeout=10)

    def launcher_setup(self, option):
        config = SimpleNamespace(root=self.root / 'session', agent_cli='codex', agent_provider=None,
                                 adaptation_source=None, agent_formal=False, formal=None)
        source = self.root / 'input.json'
        source.write_text('{}')
        self.patches.enter_context(patch.object(launcher.Configuration, 'from_dict', return_value=config))
        self.patches.enter_context(patch('services.controller.config.with_task_timeouts', side_effect=lambda c: c))
        self.patches.enter_context(patch.object(launcher, 'openrouter_key_file', return_value=None))
        self.patches.enter_context(patch.object(launcher.signal, 'signal'))
        self.patches.enter_context(patch.object(launcher.sys, 'argv', [
            'auto-research', '--config', str(source), '--provider-home', str(self.root), option]))
        deployed = {'workspace': config.root / 'agent-workspace'}
        deploy = self.patches.enter_context(patch.object(launcher, 'deploy', return_value=deployed))
        launch = self.patches.enter_context(patch.object(launcher, 'launch', side_effect=AssertionError('model launched')))
        return config, deploy, launch

    def test_prepare_only_uses_real_preflight_before_deployment(self):
        self.help_output(OLD_HELP)
        config, deploy, launch = self.launcher_setup('--prepare-only')
        with self.assertRaisesRegex(RuntimeError, 'does not support --size'):
            launcher.main()
        deploy.assert_not_called()
        launch.assert_not_called()
        lifecycle = json.loads((config.root / 'agent-lifecycle.json').read_text())
        self.assertEqual(lifecycle['status'], 'failed')
        self.assertIn('--size', lifecycle['error'])

    def test_prepare_only_accepts_supported_bwrap_without_a_model(self):
        self.help_output(NEW_HELP)
        config, deploy, launch = self.launcher_setup('--prepare-only')
        with redirect_stdout(io.StringIO()):
            launcher.main()
        deploy.assert_called_once()
        launch.assert_not_called()
        self.help.assert_called_once()
        lifecycle = json.loads((config.root / 'agent-lifecycle.json').read_text())
        self.assertEqual(lifecycle['status'], 'prepared-only')

    def test_normal_launch_also_rejects_old_bwrap_before_model_execution(self):
        self.help_output(OLD_HELP)
        _, deploy, launch = self.launcher_setup('--non-interactive')
        with self.assertRaisesRegex(RuntimeError, 'does not support --size'):
            launcher.main()
        deploy.assert_not_called()
        launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
