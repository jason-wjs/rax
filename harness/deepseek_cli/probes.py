"""Deterministic checks of the actual DSH command and native-file boundaries."""
import json
from pathlib import Path
import subprocess

from harness import agent_host
from harness.deepseek_cli import auto_research as dsh


def boundary(config, deployed):
    root, workspace = config.root, Path(deployed['workspace'])
    outside = root/'outside-boundary.txt'
    outside.write_text('Validation canary; not an experiment input')
    socket = agent_host.socket_path(root)
    home = Path(deployed[dsh.HOME_KEY])
    env = dsh.agent_environment(config, deployed, socket)
    specification = {'outside': str(outside), 'repository': str(agent_host.PROJECT/'README.md'),
        'published': str(root/'published'), 'socket': str(socket), 'workspace': str(workspace)}
    login = json.loads(Path(deployed['login']).read_text())
    if login.get('key_file'):
        specification['credential'] = login['key_file']
    code = '''
import json,os,socket
from pathlib import Path
s=json.loads(os.environ['RAX_PROBE'])
assert 'RAX_MODEL_KEY' not in os.environ
assert not Path('/dev/nvidia0').exists()
for name in ('outside','repository'):
    try: Path(s[name]).read_text()
    except OSError: pass
    else: raise AssertionError('Private path readable: '+name)
if s.get('credential'):
    try: value=Path(s['credential']).read_bytes()
    except OSError: pass
    else: assert not value, 'Credential file readable'
for path in (Path(s['outside']),Path(s['published'])/'forbidden.txt'):
    try: path.write_text('forbidden')
    except OSError: pass
    else: raise AssertionError('Protected path writable')
client=socket.socket();client.settimeout(2)
assert client.connect_ex(('1.1.1.1',443)) != 0
client.close()
client=socket.socket(socket.AF_UNIX);client.settimeout(5);client.connect(s['socket'])
client.sendall(b'{"jsonrpc":"2.0","id":11,"method":"tools/list"}\\n')
assert b'result' in client.recv(65536)
client.close()
Path(s['workspace'],'output/boundary-ok.txt').write_text('OK')
print('offline shell, hidden files, protected output, no GPU, scrubbed key, robot socket: passed')
'''
    # The runner scrubs the model credential. The fixture specification is passed
    # as a literal command argument, rather than weakening its env allowlist.
    code = code.replace("os.environ['RAX_PROBE']", repr(json.dumps(specification)))
    command = [str(agent_host.agent_python()), '-I', str(dsh.HERE/'runner.py'), str(home/'boundary.json'),
               '--', str(agent_host.agent_python()), '-I', '-c', code]
    result = subprocess.run(command, env=env, cwd=workspace, text=True, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError('DSH command boundary check failed: '+result.stderr[-1500:])
    link = workspace/'output/boundary-escape'
    link.symlink_to(outside)
    native = '''
import {pathToFileURL} from 'node:url';
import {readFileSync} from 'node:fs';
const {default:RaxFS}=await import(pathToFileURL(process.env.RAX_DSH_FS).href);
const boundary=JSON.parse(readFileSync(process.env.RAX_DSH_BOUNDARY,'utf8'));
const fs=Object.create(RaxFS.prototype);
fs.config={cwd:boundary.workspace,diffBasisMaxBytes:10485760};
fs.roots=boundary.file_read_roots;
await fs.resolve('AGENTS.md');
for(const path of JSON.parse(process.env.RAX_PROBE_DENIED)){
  let denied=false;
  try{await fs.resolve(path);}catch(e){if(e.code==='FS_SANDBOX_DENIED')denied=true;else throw e;}
  if(!denied)throw new Error('Native filesystem allowed a private path');
}
console.log('native file read boundary and symlink containment: passed');
'''
    env.update(RAX_DSH_FS=str(dsh.HERE/'filesystem.mjs'),
               RAX_PROBE_DENIED=json.dumps([str(outside), str(link), '/proc/self/environ']))
    node = json.loads(Path(deployed['login']).read_text())['node']
    try:
        result = subprocess.run([node, '--input-type=module', '-e', native], env=env, cwd=workspace,
                                text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError('DSH native file boundary check failed: '+result.stderr[-1500:])
    finally:
        link.unlink(missing_ok=True)
    return {'offline_shell': True, 'hidden_paths': True, 'readonly_recordings': True,
            'no_gpu': True, 'scrubbed_credential': True, 'robot_socket': True,
            'native_file_reads': True, 'symlink_containment': True}
