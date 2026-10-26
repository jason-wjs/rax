"""Cursor CLI's per-turn usage; cached input is separate from inputTokens."""
from pathlib import Path

from services.native_cli_usage import rows


class UsageReader:
    def read(self, home):
        turns = {}
        unidentified = 0
        for row in rows(Path(home)/'usage.jsonl'):
            if row.get('request_id'):
                turns[(row.get('session_id'), row['request_id'])] = row
            else:
                unidentified += 1
        totals = dict.fromkeys(('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
                               'output_tokens', 'total_tokens'), 0)
        complete, measured = bool(turns) and unidentified == 0, 0
        api_ms, duration_ms = 0, 0
        for row in turns.values():
            usage = row.get('usage') or {}
            api_ms += row.get('duration_api_ms') or 0
            duration_ms += row.get('duration_ms') or 0
            if not all(isinstance(usage.get(k), int) for k in ('inputTokens', 'outputTokens')):
                complete = False
                continue
            measured += 1
            cached, written = usage.get('cacheReadTokens', 0), usage.get('cacheWriteTokens', 0)
            inputs, outputs = usage['inputTokens']+cached+written, usage['outputTokens']
            totals['input_tokens'] += inputs
            totals['cached_input_tokens'] += cached
            totals['cache_write_input_tokens'] += written
            totals['output_tokens'] += outputs
            totals['total_tokens'] += inputs+outputs
        return {'token_usage': totals if measured else None,
                'total_session_usage': totals if measured else None,
                'token_usage_complete': complete, 'completed_turns': len(turns),
                'unidentified_results': unidentified,
                'model_calls': None, 'agent_steps': None,
                'duration_api_ms': api_ms, 'duration_ms': duration_ms}
