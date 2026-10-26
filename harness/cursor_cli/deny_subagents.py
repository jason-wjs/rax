"""Trusted Cursor hook enforcing this experiment's zero-subagent budget."""
import json

print(json.dumps({'permission': 'deny', 'user_message': 'This experiment uses one agent.'}))
