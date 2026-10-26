"""Messages tool images must remain visual inputs through compatible relays."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

MODULE = Path(__file__).resolve().parents[2] / 'harness/deepseek_cli/message_images.mjs'


def transform(body):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is required for the native DSH transport')
    script = """import {pathToFileURL} from 'node:url';
const {liftToolResultImages} = await import(pathToFileURL(process.argv[1]).href);
let data=''; for await (const chunk of process.stdin) data+=chunk;
console.log(JSON.stringify(liftToolResultImages(JSON.parse(data))));"""
    result = subprocess.run([node, '--input-type=module', '-e', script, str(MODULE)],
                            input=json.dumps(body), text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def test_tool_image_moves_to_user_message_without_losing_tool_result():
    image = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'AA=='}}
    body = {'messages': [{'role': 'user', 'content': [
        {'type': 'tool_result', 'tool_use_id': 'camera_1', 'is_error': False,
         'content': [{'type': 'text', 'text': 'camera frame'}, image]}]}]}
    content = transform(body)['messages'][0]['content']
    assert content[0] == {'type': 'tool_result', 'tool_use_id': 'camera_1', 'is_error': False,
                          'content': [{'type': 'text', 'text': 'camera frame'}]}
    assert content[-1] == image
    assert content[-2]['text'] == 'Images from tool result camera_1:'


def test_image_only_tool_result_stays_valid():
    image = {'type': 'image', 'source': {'type': 'url', 'url': 'https://example.invalid/frame.png'}}
    content = transform({'messages': [{'role': 'user', 'content': [
        {'type': 'tool_result', 'tool_use_id': 'camera_2', 'content': [image]}]}]})['messages'][0]['content']
    assert content[0]['tool_use_id'] == 'camera_2'
    assert content[0]['content'] == [{'type': 'text', 'text': 'Tool images attached below.'}]
    assert content[-1] == image


def test_messages_without_nested_images_do_not_change():
    body = {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]},
                         {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'x',
                                                     'content': 'plain result'}]}]}
    assert transform(body) == body
