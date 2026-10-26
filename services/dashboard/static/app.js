// Operator dashboard for auto-research sessions.
const token = new URLSearchParams(location.hash.slice(1)).get('token') || '';
const headers = {'X-Operator-Token': token};
const $ = id => document.getElementById(id);
// Live camera tiles are created per camera the simulator lists (overview first, then the wrists).
function cameraImage(camera) {
  let img = document.querySelector(`#cameras img[data-camera="${CSS.escape(camera)}"]`);
  if (!img) {
    const figure = document.createElement('figure'), caption = document.createElement('figcaption');
    img = document.createElement('img');
    img.dataset.camera = camera; img.alt = camera; img.hidden = true; figure.hidden = true;
    caption.textContent = camera;
    figure.append(img, caption);
    $('cameras').append(figure);
  }
  return img;
}
const cameraImages = () => [...document.querySelectorAll('#cameras img[data-camera]')];
const RESEARCH_TEXT = new Set(['task', 'observation_profile', 'sim_gpu', 'demonstration_context',
                               'agent_formal', 'formal_context', 'transfer_parts', 'write_handoff',
                               'adaptation_source', 'adaptation_context', 'agent_cli', 'agent_provider', 'agent_effort', 'task_rules',
                               'episode_reviews', 'expert_content']);

let data = null, busy = false, selected = '', dirty = false;
let resultsSignature = '', liveEpisode = '', liveKey = '', liveLoaded = '', artifactUrl = null;
let episodesSignature = '', episodesAt = 0, episodesRun = '';
const liveUrls = {};

// Written when the agent session ends (services/subagent_metrics.py).
const subagentRow = s => ['Subagents spawned / peak / cap',
  s ? (s.error ? 'unavailable' : `${s.spawned} / ${s.peak_concurrent} / ${s.cap ?? '—'}`) : null,
  s && !s.error ? `${s.environment_calls} environment calls, ${s.handoffs} handoffs, ${s.interleaved_calls} interleaved` : ''];
// Codex writes each subagent thread to its own rollout; the total sums them all.
const threadUsage = u => u?.threads ? `Main ${(u.main_usage?.total_tokens ?? 0).toLocaleString()} (${u.threads.main} thread${u.threads.main === 1 ? '' : 's'}); ` +
  `subagents ${(u.subagent_usage?.total_tokens ?? 0).toLocaleString()} (${u.threads.subagents})` : '';
const history = () => $('history').checked;
const form = () => $('researchForm');
const runId = () => selected;

function text(id, value, className) {
  const node = $(id);
  node.textContent = value ?? '';
  if (className !== undefined) node.className = className;
}

async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {...headers, ...(options.headers || {})}, cache: 'no-store'});
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `${response.status} ${response.statusText}`);
  return body;
}

function post(path, value) {
  return api(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(value)});
}

function query(extra = {}) {
  const values = {...extra, ...(history() ? {history: '1'} : {})};
  return new URLSearchParams(Object.entries(values).filter(([, value]) => value !== '' && value != null));
}

function runQuery(extra = {}) {
  return query({run_id: runId(), ...extra});
}

function syncSelect(select, rows, value) {
  const signature = JSON.stringify(rows);
  if (select.dataset.signature !== signature) {
    select.replaceChildren(...rows.map(([id, label]) => new Option(label, id)));
    select.dataset.signature = signature;
  }
  select.value = value;
}

function percent(value) { return value == null ? '—' : (100*value).toFixed(1)+'%'; }

function stats(rows) {
  $('stats').replaceChildren(...rows.map(([label, value, title]) => {
    const div = document.createElement('div'), k = document.createElement('div'), v = document.createElement('div');
    div.className = 'stat'; k.className = 'k'; v.className = 'v';
    k.textContent = label; v.textContent = value ?? '—';
    if (title) v.title = title;
    div.append(k, v);
    return div;
  }));
}

// ---- setup forms -----------------------------------------------------------

function formValues(element, textFields) {
  const values = {};
  for (const input of element.elements) {
    if (!input.name) continue;
    values[input.name] = textFields.has(input.name) ? input.value.trim() : Number(input.value);
  }
  return values;
}

function fillForm(element, values) {
  for (const input of element.elements) if (input.name && input.name in values) input.value = values[input.name];
}

function loadForm() {
  if (!data) return;
  for (const select of document.querySelectorAll('.taskSelect')) {
    if (!select.options.length) select.replaceChildren(...data.tasks.map(task =>
      new Option(benchmarkInfo(task).task_tag ? `${task} · ${benchmarkInfo(task).task_tag}` : task, task)));
  }
  for (const select of document.querySelectorAll('.demoSelect')) {
    if (!select.options.length) select.replaceChildren(...(data.demonstration_contexts || []).map(kind => new Option(kind, kind)));
  }
  if (!dirty) {
    form().elements.agent_cli.value = data.profile.agent_cli;
    updateAgentProviders(data.profile.agent_provider);
    fillForm(form(), data.profile);
  } else updateAgentProviders();
  formBenchmark = benchmarkOf(form().elements.task.value);
  applyBenchmark(false);
  $('prepare').disabled = false;
  updateTimeouts();
}

function updateAgentProviders(preferred) {
  const elements = form().elements, providers = data?.agent_providers?.[elements.agent_cli.value] || [];
  const selected = preferred ?? elements.agent_provider.value;
  syncSelect(elements.agent_provider, providers.map(value => [value, value]),
    providers.includes(selected) ? selected : providers[0]);
}

let formBenchmark = null;
const benchmarkValues = {};  // Last edited per-benchmark fields while the page is open.

function benchmarkOf(task) {
  return data?.benchmarks?.[task] || '';
}

function benchmarkInfo(task) {
  return data?.benchmark_info?.[benchmarkOf(task)] || {};
}

// Each benchmark has its own layout collections and fixed fields; switching the task to
// another benchmark loads that benchmark's defaults into the editable fields.
function applyBenchmark(switched, previous) {
  const elements = form().elements, benchmark = benchmarkOf(elements.task.value);
  if (switched) {
    const fields = Object.keys(data?.benchmark_defaults?.[benchmark] || {});
    if (previous) benchmarkValues[previous] = Object.fromEntries(fields.map(name => [name, elements[name].value]));
    fillForm(form(), benchmarkValues[benchmark] || data?.benchmark_defaults?.[benchmark] || {});
  }
  // Fields that do not apply to this benchmark are fixed and disabled, with the reason as a tooltip.
  const fixed = data?.benchmark_fixed?.[benchmark] || {};
  // Include fields fixed earlier on this page: a restarted server may no longer fix them.
  const allFixed = new Set(Object.values(data?.benchmark_fixed || {}).flatMap(fields => Object.keys(fields)));
  for (const input of elements) if (input.dataset.benchmarkFixed) allFixed.add(input.name);
  for (const name of allFixed) {
    const input = elements[name];
    if (!input) continue;
    const label = input.closest('label');
    if (label.dataset.title === undefined) label.dataset.title = label.title;
    input.disabled = name in fixed;
    if (name in fixed) input.dataset.benchmarkFixed = '1'; else delete input.dataset.benchmarkFixed;
    label.title = name in fixed ? fixed[name].reason : label.dataset.title;
    if (name in fixed) input.value = fixed[name].value;
  }
}

function updateTimeouts() {
  const task = form().elements.task.value, budget = data?.task_timeouts?.[task];
  const note = benchmarkInfo(task).layout_note, layouts = note ? ` · ${note}` : '';
  text('timeouts', budget ? `${budget.native_step_limit} native steps · ${budget.episode_seconds ? Math.round(budget.episode_seconds/60) + ' min' : 'no wall-clock limit'} per episode; no session time limit.${layouts}` : '');
}

// ---- live cameras ----------------------------------------------------------

function clearLive() {
  for (const img of cameraImages()) {
    const camera = img.dataset.camera;
    img.hidden = true; img.closest('figure').hidden = true; img.removeAttribute('src');
    if (liveUrls[camera]) URL.revokeObjectURL(liveUrls[camera]);
    delete liveUrls[camera];
  }
  liveKey = liveLoaded = '';
}

function liveView() {
  // What to show and where to fetch each camera.
  const live = data?.selected?.live_observation;
  if (!live) return null;
  return {episode: live.native_run, step: live.step_id, available: live.available, active: live.active,
    mode: live.episode_mode, limit: live.episode_step_limit, updated: live.updated_at_unix_s, id: live.episode_id,
    reward: live.reward || {}, cameras: live.cameras || [],
    url: camera => '/api/research/frame?'+query({run_id: runId(), native_run: live.native_run, step_id: live.step_id, camera})};
}

async function renderLive() {
  const live = liveView();
  const episode = JSON.stringify([runId(), live?.episode]);
  if (episode !== liveEpisode) { clearLive(); liveEpisode = episode; }
  if (!live?.available) {
    clearLive();
    text('liveStatus', 'No native observation yet.', 'muted');
    text('liveMetrics', '');
    return;
  }
  const age = live.updated ? Math.max(0, Math.floor(Date.now()/1000 - live.updated)) : null;
  text('liveStatus', `${live.active ? 'Simulator live' : 'Simulator stopped · last saved observation'} · ${live.mode || 'unknown'} · step ${live.step} / ${live.limit ?? '—'}${age === null ? '' : ` · saved ${age}s ago`}`, live.active ? 'good' : 'muted');
  // A simulator may report terminated/truncated instead of a native end flag.
  const reward = live.reward, ended = 'terminated' in reward ? Boolean(reward.terminated || reward.truncated) : undefined;
  text('liveMetrics', `Episode ${live.id || '—'} · native success ${reward.native_success ?? '—'} · score ${reward.native_score_percent ?? '—'} · terminal ${reward.native_end_flag ?? ended ?? '—'}`, 'mono muted');
  const cameras = live.cameras;
  const key = JSON.stringify([episode, live.step, cameras]);
  if (key === liveKey || key === liveLoaded) return;
  liveKey = key;
  try {
    const frames = await Promise.all(cameras.map(async camera => {
      const response = await fetch(live.url(camera), {headers, cache: 'no-store'});
      if (!response.ok) throw new Error('Live frame unavailable; retrying on refresh.');
      return [camera, await response.blob()];
    }));
    if (liveKey !== key) return;
    for (const camera of new Set([...cameraImages().map(img => img.dataset.camera), ...cameras])) {
      if (liveUrls[camera]) URL.revokeObjectURL(liveUrls[camera]);
      delete liveUrls[camera];
      const frame = frames.find(([name]) => name === camera), img = cameraImage(camera);
      img.hidden = !frame;
      img.closest('figure').hidden = !frame;
      if (frame) { liveUrls[camera] = URL.createObjectURL(frame[1]); img.src = liveUrls[camera]; }
      else img.removeAttribute('src');
    }
    liveLoaded = key;
  } catch (error) {
    if (liveKey === key) text('liveStatus', error.message, 'muted');
  } finally {
    if (liveKey === key) liveKey = '';
  }
}

// ---- episode videos --------------------------------------------------------

function resetVideo() {
  $('video').pause(); $('video').removeAttribute('src'); $('video').hidden = true;
  $('download').hidden = true; text('playing', '');
}

function playEpisode(row, index) {
  const url = '/api/research/video?'+runQuery({episode: row.episode});
  const video = $('video');
  video.hidden = false; video.src = url;
  video.playbackRate = Number($('speed').value);
  video.play().catch(() => {});
  $('download').href = url;
  $('download').download = `${runId()}-${index}-${row.mode}.mp4`;
  $('download').hidden = false;
  text('playing', `Episode ${index} · ${row.mode}${row.formal_episode_index ? ' '+row.formal_episode_index : ''} · ${row.episode_id || ''}`, 'muted');
}

const REVIEW_LABELS = {ready: 'Read', error: 'Failed', unavailable: 'No ground truth', budget_exhausted: 'Over budget'};

function reviewCell(row, index) {
  const td = document.createElement('td');
  const status = row.review?.status;
  if (status !== 'ready') {
    td.className = 'muted';
    td.textContent = status ? REVIEW_LABELS[status] || status : '—';
    return td;
  }
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = `Read ($${(row.review.cost_usd || 0).toFixed(3)})`;
  button.addEventListener('click', () => showReview(row, index));
  td.append(button);
  return td;
}

async function showReview(row, index) {
  try {
    const record = await api('/api/research/review?'+runQuery({episode_id: row.episode_id}));
    $('reviewTitle').textContent = `Episode ${index} review (${row.episode_id})`;
    const missed = (record.report?.missed || []).map(item => `- ${item.condition}${item.geometry ? ': '+item.geometry : ''}`);
    // A report that carries its own text summary shows it; otherwise its missed goal conditions.
    const truth = record.report?.summary ? [record.report.summary]
      : (missed.length ? ['Missed:', ...missed] : ['No missed conditions']);
    const warnings = (record.report?.warnings || []).map(item => `Warning: ${item}`);
    $('reviewText').textContent = [
      'Agent sees:', '', record.agent_review || '(none)', '',
      '--- Operator only: ground-truth report ---', '', ...truth, ...(warnings.length ? ['', ...warnings] : []),
      '', JSON.stringify(record.report?.private || {}, null, 1)].join('\n');
    $('reviewPanel').hidden = false;
  } catch (error) {
    text('message', error.message, 'bad');
  }
}

async function refreshReviews() {
  if (!runId()) { $('reviewControls').hidden = true; return; }
  try {
    const value = await api('/api/research/reviews?'+runQuery());
    $('reviewControls').hidden = !value.enabled;
    $('reviewsInfo').textContent = `Episode reviews on · analyst spend $${value.spent_usd.toFixed(2)} of $${value.budget_usd.toFixed(0)}`;
  } catch (error) {
    $('reviewControls').hidden = true;
  }
}

async function refreshEpisodes(force = false) {
  // Episode lists change slowly; poll every 10 s, or at once on a new selection.
  const key = JSON.stringify([runId(), history()]);
  if (!runId()) { $('episodes').replaceChildren(); episodesSignature = ''; episodesRun = key; resetVideo(); return; }
  if (!force && key === episodesRun && Date.now() - episodesAt < 10000) return;
  refreshReviews();
  let rows;
  try {
    rows = await api('/api/research/episodes?'+runQuery());
  } catch (error) {
    text('message', error.message, 'bad');
    return;
  }
  if (key !== JSON.stringify([runId(), history()])) return;
  if (key !== episodesRun) resetVideo();
  episodesRun = key; episodesAt = Date.now();
  const signature = JSON.stringify(rows);
  if (signature === episodesSignature) return;
  episodesSignature = signature;
  $('episodes').replaceChildren(...rows.map((row, i) => {
    const tr = document.createElement('tr');
    const result = row.live ? 'running' : row.native_success === true ? 'success' : row.native_success === false ? 'not successful' : 'unknown';
    const cells = [i + 1, row.mode + (row.formal_episode_index ? ' '+row.formal_episode_index : ''),
      (row.episode_id || '—').slice(0, 12), `${row.steps ?? '—'}${row.step_limit ? ' / '+row.step_limit : ''}`,
      result, row.native_score == null ? '—' : (100*row.native_score).toFixed(1)].map(value => {
        const td = document.createElement('td'); td.textContent = value; return td; });
    cells[4].className = result === 'success' ? 'good' : result === 'not successful' ? 'bad' : 'muted';
    const td = document.createElement('td'), button = document.createElement('button');
    button.type = 'button';
    // The simulator finalizes an episode's MP4 when the episode closes.
    button.textContent = row.live ? 'Recording…' : row.video ? `Play (${(row.video_bytes / 1048576).toFixed(1)} MB)` : 'No video';
    button.disabled = row.live || !row.video;
    button.addEventListener('click', () => playEpisode(row, i + 1));
    td.append(button);
    tr.append(...cells, td, reviewCell(row, i + 1));
    return tr;
  }));
  if (!rows.length) {
    const tr = document.createElement('tr'), td = document.createElement('td');
    td.colSpan = 8; td.className = 'muted'; td.textContent = 'No simulator episodes recorded yet.';
    tr.append(td); $('episodes').append(tr);
  }
}

// ---- research sessions -----------------------------------------------------

function renderResults(session) {
  const signature = JSON.stringify([session?.run_id, session?.results]);
  if (signature === resultsSignature) return;
  resultsSignature = signature;
  $('results').replaceChildren(...[...(session?.results || [])].reverse().map(row => {
    const tr = document.createElement('tr');
    const cells = [
      `${row.mode} · ${row.formal_episode_index ?? row.episode ?? row.episode_id ?? '—'}`,
      (row.bundle || '—').slice(0, 12),
      row.task_complete === true ? 'true' : row.task_complete === false ? 'false' : 'unknown',
      `${row.reason || '—'} / exit ${row.returncode ?? '—'}${row.error_type ? ' / '+row.error_type : ''}`,
    ].map(value => { const td = document.createElement('td'); td.textContent = value; return td; });
    cells[2].className = row.task_complete === true ? 'good' : row.task_complete === false ? 'bad' : 'muted';
    const td = document.createElement('td'), details = document.createElement('details');
    const summary = document.createElement('summary'), pre = document.createElement('pre');
    summary.textContent = 'Result'; pre.textContent = JSON.stringify(row, null, 2);
    details.append(summary, pre);
    const artifacts = [['stdout', row.artifacts?.stdout], ['stderr', row.artifacts?.stderr],
      ...(row.artifacts?.final_images || []).map((frame, index) => ['final image '+(index+1), frame.path])];
    for (const [label, path] of artifacts) {
      if (!path) continue;
      const button = document.createElement('button');
      button.type = 'button'; button.textContent = label;
      button.addEventListener('click', () => viewArtifact(session.run_id, path));
      details.append(button);
    }
    td.append(details);
    tr.append(...cells, td);
    return tr;
  }));
}

function agentLabel(agent) {
  if (!agent) return null;
  const effort = agent.effort ? ` · effort ${agent.effort}` : '';
  if (agent.cli === 'codex') return agent.cli + effort;
  const account = agent.provider === 'claude-login'
    ? ` · ${agent.account || 'no experiment token'}${agent.pending ? ' (at launch)' : ''}` : '';
  return `${agent.cli} · ${agent.provider}${account}${effort}`;
}

function renderResearch() {
  const session = data.selected;
  selected = session?.run_id || '';
  syncSelect($('run'), data.sessions.length ? data.sessions.map(row => [row.run_id, `${row.task} · ${row.run_id} · ${row.status}`]) : [['', 'No sessions']], runId());
  const batch = session?.formal_batch, score = batch?.score_summary, gemini = session?.gemini;
  const agentScore = session?.agent_formal_batch?.score_summary;
  const scoreText = summary => summary.average_score_percent != null ? summary.average_score_percent.toFixed(1)+'%'
    : `pending · ${summary.recorded_score_count} recorded` +
      (summary.recorded_average_score_percent != null ? ` (${summary.recorded_average_score_percent.toFixed(1)}% so far)` : '');
  const scoreLabel = benchmarkInfo(session?.task).score_label || 'Formal average score';
  const rows = [
    ['Task', session?.task], ['Lifecycle', session?.status], ['Stage', session?.stage],
    ['Manual success', session ? String(session.manual_success) : null],
    ['Exploration used', session ? `${session.exploration_used} / ${session.exploration_limit}` : null],
    ['Formal success', batch ? `${batch.success_count} / ${batch.episode_count} (${percent(batch.success_rate)})` : null],
    [scoreLabel, score ? scoreText(score) : null],
    ['Agent formal success', session?.agent_formal_batch ? `${session.agent_formal_batch.success_count} / ${session.agent_formal_batch.episode_count} (${percent(session.agent_formal_batch.success_rate)})` : null,
      session?.agent_checkpoint ? `Transferred ${Object.entries(session.agent_checkpoint.parts || {}).filter(([, v]) => v).map(([k]) => k).join(', ') || 'nothing'}` +
        (Object.keys(session.agent_checkpoint.excluded || {}).length ? `; over cap: ${Object.keys(session.agent_checkpoint.excluded).join(', ')}` : '') +
        (session.agent_checkpoint.compaction ? `; compaction ${session.agent_checkpoint.compaction.status}` : '') : ''],
    [`Agent ${scoreLabel[0].toLowerCase()}${scoreLabel.slice(1)}`, agentScore ? scoreText(agentScore) : null],
    ['Agent tokens', session?.usage?.total_session_usage?.total_tokens?.toLocaleString(), threadUsage(session?.usage)],
    ['Gemini spend', gemini ? (gemini.blocked ? 'Blocked' : `$${gemini.reported_usd.toFixed(3)}${gemini.limit_usd == null ? '' : ` / $${gemini.limit_usd}`}`) : null,
      gemini ? `Reserved $${gemini.reserved_usd.toFixed(4)}; ${gemini.remaining_usd == null ? 'no session limit' : `remaining $${gemini.remaining_usd.toFixed(3)}`}. Codex costs are separate.` : ''],
    ['Agent', agentLabel(session?.agent)],
    subagentRow(session?.subagents),
  ];
  stats(rows);
  text('command', session?.command || 'No session selected.', session ? '' : 'muted');
  $('launch').disabled = !session?.launchable;
  $('stop').disabled = !session?.alive;
  $('resume').disabled = !session?.resumable;
  text('launchNote', 'Launch runs the prepared configuration headlessly, once. To change a setup, prepare a new session.', 'muted');
  text('paths', session ? `Workspace: ${session.workspace}\nArtifacts: ${session.artifacts}` : '');
  text('agentLog', session?.agent_log || 'No captured log. Terminal launches print to their terminal.');
  text('frontendLog', session?.frontend_log || 'No frontend log yet.');
  text('configuration', JSON.stringify({configuration: session?.configuration, bundles: session?.bundles,
    rehearsed_bundles: session?.rehearsed_bundles, usage: session?.usage}, null, 2));
  renderResults(session);
}

async function viewArtifact(run, path) {
  try {
    const response = await fetch('/api/research/artifact?'+query({run_id: run, path}), {headers, cache: 'no-store'});
    if (!response.ok) throw new Error((await response.json()).error);
    if (artifactUrl) { URL.revokeObjectURL(artifactUrl); artifactUrl = null; }
    const image = response.headers.get('Content-Type').startsWith('image/');
    $('artifactImage').hidden = !image; $('artifactText').hidden = image;
    if (image) { artifactUrl = URL.createObjectURL(await response.blob()); $('artifactImage').src = artifactUrl; }
    else text('artifactText', await response.text());
    text('artifactPath', `${run} / ${path}`, 'mono');
    $('artifactViewer').open = true;
  } catch (error) {
    text('message', error.message, 'bad');
  }
}

// ---- main loop -------------------------------------------------------------

function render() {
  loadForm();
  renderResearch();
  refreshEpisodes();
  renderLive();
}

async function refresh() {
  if (busy) return;
  busy = true;
  const requested = [runId(), history()];
  try {
    const value = await api('/api/research/state?'+query({run_id: runId()}));
    if (JSON.stringify(requested) === JSON.stringify([runId(), history()])) { data = value; render(); }
  } catch (error) {
    text('message', error.message, 'bad');
  } finally {
    busy = false;
  }
}

function switchView() {
  data = null; resultsSignature = ''; episodesSignature = ''; episodesRun = '';
  clearLive(); resetVideo();
  refresh();
}

$('run').addEventListener('change', () => { selected = $('run').value; switchView(); });
$('history').addEventListener('change', () => { selected = ''; switchView(); });
$('refresh').addEventListener('click', refresh);
$('researchForm').addEventListener('input', event => {
  dirty = true;
  if (event.target.name === 'agent_cli') updateAgentProviders();
  if (event.target.name === 'task' && benchmarkOf(event.target.value) !== formBenchmark) {
    const previous = formBenchmark;
    formBenchmark = benchmarkOf(event.target.value);
    applyBenchmark(true, previous);
  }
  updateTimeouts();
  text('message', 'Unsaved setup edits; Prepare creates a new session and leaves the selected one unchanged.', 'muted');
});
$('researchForm').addEventListener('submit', async event => {
  event.preventDefault();
  $('prepare').disabled = true;
  try {
    const prepared = await post('/api/research/prepare', formValues($('researchForm'), RESEARCH_TEXT));
    selected = prepared.run_id; dirty = false;
    text('message', 'Configuration prepared; nothing is running yet. Launch it below or copy the terminal command.', 'good');
    switchView();
  } catch (error) {
    text('message', error.message, 'bad');
  } finally {
    $('prepare').disabled = false;
  }
});
$('resume').addEventListener('click', async () => {
  const session = data?.selected;
  if (!session?.resumable || !confirm(`Resume the ${session.task} session? It continues the agent's own conversation in the same workspace, with its remaining Gemini budget.`)) return;
  try {
    text('message', (await post('/api/research/resume', {run_id: session.run_id})).status, 'good');
  } catch (error) { text('message', error.message, 'bad'); }
});

$('launch').addEventListener('click', async () => {
  if (!data) return;
  const session = data.selected;
  if (!session || !confirm(`Launch the paid agent session for ${session.task}? It uses the prepared configuration, not unsaved form edits.`)) return;
  try {
    text('message', (await post('/api/research/launch', {run_id: session.run_id})).status, 'good');
  } catch (error) {
    text('message', error.message, 'bad');
  } finally {
    await refresh();
  }
});
$('stop').addEventListener('click', async () => {
  if (!runId()) return;
  if (!confirm('Stop this session and close its environment? Budgets are not restored and no formal retry is permitted.')) return;
  try {
    text('message', (await post('/api/research/stop', {run_id: runId()})).status, 'good');
  } catch (error) {
    text('message', error.message, 'bad');
  }
  await refresh();
});
document.addEventListener('click', async event => {
  const source = event.target.closest('[data-copy]');
  if (!source) return;
  try { await navigator.clipboard.writeText($(source.dataset.copy).textContent); text('message', 'Copied.', 'good'); }
  catch (_) { text('message', 'Clipboard unavailable; select the text and copy it.', 'muted'); }
});
$('reviewClose').addEventListener('click', () => { $('reviewPanel').hidden = true; });
$('speed').addEventListener('change', () => { $('video').playbackRate = Number($('speed').value); });

// #history=1 opens the dashboard with earlier runs listed.
$('history').checked = new URLSearchParams(location.hash.slice(1)).get('history') === '1';
if (!token) text('message', 'Missing operator token: open the URL printed by the dashboard server.', 'bad');
fetch('/api/session', {method: 'POST', headers}).catch(() => {});  // Cookie for <video> requests.
refresh();
setInterval(refresh, 1000);
