/* RLLM Dashboard — client-side logic */

// ── Constants ────────────────────────────────────────────────────────────

const NO_JOB = '\u2014'; // em-dash sentinel for "no SLURM job"

// ── State ────────────────────────────────────────────────────────────────

const state = {
  experiments: [],
  slurmJobs: [],
  metadata: { dataset_categories: {}, model_tabs: [] },
  filters: { datasetCategory: null, model: null, status: 'All' },
  selectedExperiment: null,
  refreshTimer: null,
};

// ── DOM refs ─────────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const pivotGrid = $('#pivot-grid');
const emptyState = $('#empty-state');
const slurmTbody = $('#slurm-tbody');
const detailPanel = $('#detail-panel');
const actionOutput = $('#action-output');
const launchOutput = $('#launch-output');

// ── Fetch helpers ────────────────────────────────────────────────────────

async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  return res.json();
}

// ── Data loading ─────────────────────────────────────────────────────────

async function loadExperiments() {
  const data = await api('GET', '/api/experiments');
  state.experiments = data.experiments || [];
  state.slurmJobs = data.slurm_jobs || [];
  state.metadata = data.metadata || state.metadata;
  buildTabs();
  populateEvalDatasets();
  renderPivotGrid();
  renderSlurmTable();
  // If detail panel is open, refresh its content
  if (state.selectedExperiment && detailPanel.style.display !== 'none') {
    await loadExperimentDetail(state.selectedExperiment);
  }
}

// ── Tab building ─────────────────────────────────────────────────────────

let tabsBuilt = false;

function buildTabs() {
  if (tabsBuilt) return;
  tabsBuilt = true;

  const dsContainer = $('#dataset-tabs');
  const modelContainer = $('#model-tabs');
  const cats = state.metadata.dataset_categories || {};
  const models = state.metadata.model_tabs || [];

  // Dataset category tabs — first one active by default
  const catKeys = Object.keys(cats);
  catKeys.forEach((cat, i) => {
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (i === 0 ? ' active' : '');
    btn.dataset.tab = 'dataset';
    btn.dataset.value = cat;
    btn.textContent = cat;
    btn.addEventListener('click', () => selectTab('dataset', cat));
    dsContainer.appendChild(btn);
  });
  if (catKeys.length > 0) {
    state.filters.datasetCategory = catKeys[0];
  }

  // Model tabs — first one active by default
  models.forEach((m, i) => {
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (i === 0 ? ' active' : '');
    btn.dataset.tab = 'model';
    btn.dataset.value = m;
    btn.textContent = m;
    btn.addEventListener('click', () => selectTab('model', m));
    modelContainer.appendChild(btn);
  });
  if (models.length > 0) {
    state.filters.model = models[0];
  }
}

function selectTab(type, value) {
  if (type === 'dataset') {
    state.filters.datasetCategory = value;
    $$('[data-tab="dataset"]').forEach(b => {
      b.classList.toggle('active', b.dataset.value === value);
    });
  } else {
    state.filters.model = value;
    $$('[data-tab="model"]').forEach(b => {
      b.classList.toggle('active', b.dataset.value === value);
    });
  }
  renderPivotGrid();
}

// Status filter buttons (static in HTML)
$$('[data-filter="status"]').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('[data-filter="status"]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.filters.status = btn.dataset.value;
    renderPivotGrid();
  });
});

// ── Filtering ────────────────────────────────────────────────────────────

function matchesFilters(exp) {
  // Dataset category filter
  const cats = state.metadata.dataset_categories || {};
  const activeCat = state.filters.datasetCategory;
  if (activeCat && cats[activeCat]) {
    const datasets = cats[activeCat];
    if (!datasets.some(d => d.toLowerCase() === exp.dataset.toLowerCase())) {
      return false;
    }
  }

  // Model filter
  if (state.filters.model && exp.model.toLowerCase() !== state.filters.model.toLowerCase()) {
    return false;
  }

  // Status filter
  if (state.filters.status !== 'All' && exp.status !== state.filters.status) {
    return false;
  }

  return true;
}

// ── Pivot Grid Rendering ─────────────────────────────────────────────────

function renderPivotGrid() {
  const filtered = state.experiments.filter(matchesFilters);

  if (filtered.length === 0) {
    pivotGrid.innerHTML = '';
    emptyState.style.display = 'block';
    return;
  }
  emptyState.style.display = 'none';

  // Collect unique workflows (columns) and policies (rows)
  const workflowSet = new Set();
  const policySet = new Set();
  for (const exp of filtered) {
    workflowSet.add(exp.workflow);
    policySet.add(exp.policy);
  }

  const workflows = [...workflowSet].sort();
  const policies = [...policySet].sort();

  // Build lookup: policy -> workflow -> [experiments]
  const lookup = {};
  for (const exp of filtered) {
    if (!lookup[exp.policy]) lookup[exp.policy] = {};
    if (!lookup[exp.policy][exp.workflow]) lookup[exp.policy][exp.workflow] = [];
    lookup[exp.policy][exp.workflow].push(exp);
  }

  const numCols = workflows.length;
  pivotGrid.style.gridTemplateColumns = `160px repeat(${numCols}, 1fr)`;
  pivotGrid.className = 'pivot-grid';

  let html = '';

  // Header row: corner + workflow headers
  html += '<div class="pivot-corner">Policy \\ Workflow</div>';
  for (const wf of workflows) {
    html += `<div class="pivot-header">${esc(wf)}</div>`;
  }

  // Data rows
  for (const policy of policies) {
    html += `<div class="pivot-row-header">${esc(policy)}</div>`;
    for (const wf of workflows) {
      const exps = (lookup[policy] && lookup[policy][wf]) || [];
      if (exps.length === 0) {
        html += '<div class="pivot-cell empty">&mdash;</div>';
      } else {
        html += '<div class="pivot-cell">';
        for (const exp of exps) {
          const pct = exp.total_steps > 0 ? Math.min(100, Math.round(exp.steps / exp.total_steps * 100)) : 0;
          const statusClass = exp.status.toLowerCase();
          const isSelected = state.selectedExperiment === exp.name;

          // Eval best scores
          let evalHtml = '';
          const eb = exp.eval_best || {};
          const entries = Object.entries(eb);
          if (entries.length) {
            evalHtml = entries.map(([ds, acc]) =>
              `<span class="eval-score">${esc(ds)}: ${(acc * 100).toFixed(1)}%</span>`
            ).join(' ');
          }
          if (exp.eval_status && exp.eval_status.includes('Evaluating')) {
            evalHtml = '&#128260; ' + evalHtml;
          }

          html += `<div class="pivot-cell-entry${isSelected ? ' selected' : ''}" data-name="${esc(exp.name)}">`;
          html += `<div class="cell-name">${esc(exp.name)}</div>`;
          html += '<div class="cell-status">';
          html += `<div class="progress-inline">`;
          html += `<div class="progress-bar-wrap"><div class="progress-bar-fill ${statusClass}" style="width:${pct}%"></div></div>`;
          html += `<span class="progress-label">${exp.steps}/${exp.total_steps}</span>`;
          html += `</div>`;
          html += `<span class="badge badge-${statusClass}">${esc(exp.status)}</span>`;
          if (exp.gpu_count) html += `<span class="gpu-badge">${exp.gpu_count} GPU</span>`;
          html += '</div>';
          if (evalHtml) {
            html += `<div class="cell-eval">${evalHtml}</div>`;
          }
          html += '</div>';
        }
        html += '</div>';
      }
    }
  }

  pivotGrid.innerHTML = html;
}

// Persistent event delegation for pivot cell clicks
pivotGrid.addEventListener('click', (e) => {
  const entry = e.target.closest('.pivot-cell-entry');
  if (!entry) return;
  const name = entry.dataset.name;
  if (name) selectExperiment(name);
});

// ── Experiment Selection (Inline Detail Panel) ───────────────────────────

async function selectExperiment(name) {
  state.selectedExperiment = name;
  // Update pivot cell highlights
  $$('.pivot-cell-entry').forEach(el => {
    el.classList.toggle('selected', el.dataset.name === name);
  });
  // Show detail panel
  detailPanel.style.display = 'block';
  actionOutput.classList.remove('visible');
  await loadExperimentDetail(name);
  // Scroll panel into view
  detailPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeDetailPanel() {
  detailPanel.style.display = 'none';
  state.selectedExperiment = null;
  $$('.pivot-cell-entry').forEach(el => el.classList.remove('selected'));
}

async function loadExperimentDetail(name) {
  try {
    const data = await api('GET', `/api/experiment/${encodeURIComponent(name)}`);
    if (data.error) {
      $('#panel-title').textContent = 'Not Found';
      return;
    }
    const exp = data.experiment;
    const ev = data.eval;

    $('#panel-title').textContent = exp.name;

    // Metadata grid
    const meta = [
      ['Workflow', exp.workflow],
      ['Policy', exp.policy],
      ['Model', exp.model],
      ['Dataset', exp.dataset],
      ['GPUs', exp.gpu_count != null ? String(exp.gpu_count) : '—'],
      ['Status', exp.status],
      ['SLURM Job', exp.slurm_job],
      ['SLURM State', exp.slurm_state],
      ['Time', exp.time],
      ['Node', exp.node],
      ['WandB Run', exp.wandb_run],
      ['Eval Status', exp.eval_status],
      ['Eval Job', exp.eval_job],
    ];
    $('#panel-meta').innerHTML = meta.map(([k, v]) =>
      `<span class="meta-key">${esc(k)}</span><span class="meta-val">${esc(v)}</span>`
    ).join('');

    // Progress bar
    const pct = exp.total_steps > 0 ? Math.min(100, Math.round(exp.steps / exp.total_steps * 100)) : 0;
    const statusClass = exp.status.toLowerCase();
    const bar = $('#panel-progress-bar');
    bar.style.width = pct + '%';
    bar.className = 'progress-bar-large';
    if (statusClass === 'finished') bar.style.background = 'var(--green)';
    else if (statusClass === 'running') bar.style.background = 'var(--blue)';
    else bar.style.background = 'var(--amber)';
    $('#panel-progress-text').textContent = `${exp.steps} / ${exp.total_steps} (${pct}%)`;

    // Conditional assign/cancel job display
    const hasJob = exp.slurm_job && exp.slurm_job !== NO_JOB;
    const assignRow = $('#assign-job-row');
    const cancelRow = $('#cancel-job-row');
    if (assignRow) assignRow.style.display = hasJob ? 'none' : 'flex';
    if (cancelRow) cancelRow.style.display = hasJob ? 'flex' : 'none';
    if (!hasJob) populateAssignJobDropdown();

    // Pre-fill resume GPUs from experiment (reset to default 2 if unavailable)
    const resumeGpus = $('#resume-n-gpus');
    if (resumeGpus) resumeGpus.value = exp.gpu_count != null ? exp.gpu_count : 2;

    // Eval cards
    renderEvalCards(ev);
  } catch (err) {
    console.error('loadExperimentDetail error:', err);
  }
}

// ── Eval Cards ───────────────────────────────────────────────────────────

function renderEvalCards(ev) {
  const container = $('#panel-eval');
  if (!ev || !ev.rows || ev.rows.length === 0) {
    container.innerHTML = '<p style="color:var(--text-secondary);font-size:0.85rem">No evaluation results.</p>';
    return;
  }

  const datasets = ev.datasets || [];
  const best = ev.best || {};

  if (datasets.length === 0) {
    container.innerHTML = '<p style="color:var(--text-secondary);font-size:0.85rem">No evaluation results.</p>';
    return;
  }

  let html = '<div class="eval-cards">';

  for (const ds of datasets) {
    // Filter rows that have data for this dataset
    const dsRows = ev.rows.filter(row => row.scores[ds] != null);
    if (dsRows.length === 0) continue;

    const bestInfo = best[ds];
    const bestAcc = bestInfo ? bestInfo.accuracy : null;
    const bestStep = bestInfo ? bestInfo.step : null;

    html += '<div class="eval-card">';

    // Card header
    html += '<div class="eval-card-header">';
    html += `<span class="eval-dataset-name">${esc(ds)}</span>`;
    if (bestInfo) {
      html += `<span class="eval-best-badge">Best: ${(bestAcc * 100).toFixed(1)}% (step ${bestStep})</span>`;
    }
    html += '</div>';

    // Card body: mini table
    html += '<div class="eval-card-body"><table><thead><tr>';
    html += '<th>Step</th><th>Mode</th><th>N</th><th>Accuracy</th>';
    html += '</tr></thead><tbody>';

    for (const row of dsRows) {
      const acc = row.scores[ds];
      const isBest = bestAcc !== null && acc === bestAcc;
      html += `<tr class="${isBest ? 'best-row' : ''}">`;
      html += `<td>${row.step}</td>`;
      html += `<td>${esc(row.mode)}</td>`;
      html += `<td>${row.n_rollouts}</td>`;
      html += `<td>${(acc * 100).toFixed(1)}%</td>`;
      html += '</tr>';
    }

    html += '</tbody></table></div>';
    html += '</div>'; // .eval-card
  }

  html += '</div>'; // .eval-cards
  container.innerHTML = html;
}

// ── Assign Job Dropdown ──────────────────────────────────────────────────

function populateAssignJobDropdown() {
  const sel = $('#assign-job-select');
  sel.innerHTML = '';
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = '-- Select a job --';
  sel.appendChild(placeholder);
  for (const j of state.slurmJobs) {
    const opt = document.createElement('option');
    opt.value = j.job_id;
    opt.textContent = `${j.job_id} - ${j.name} (${j.state})`;
    sel.appendChild(opt);
  }
}

// ── Utility ──────────────────────────────────────────────────────────────

function esc(s) {
  if (s == null) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function getSelectedExp() {
  if (!state.selectedExperiment) return null;
  return state.experiments.find(e => e.name === state.selectedExperiment) || null;
}

function inferTaskType(dataset) {
  return dataset === 'deepcoder' ? 'deepcoder' : 'math';
}

// ── SLURM table ──────────────────────────────────────────────────────────

function renderSlurmTable() {
  slurmTbody.innerHTML = state.slurmJobs.map(j => `<tr>
    <td>${esc(j.job_id)}</td>
    <td>${esc(j.partition)}</td>
    <td>${esc(j.name)}</td>
    <td>${esc(j.state)}</td>
    <td>${esc(j.time)}</td>
    <td>${esc(j.nodes)}</td>
    <td>${esc(j.nodelist)}</td>
  </tr>`).join('');
}

// ── Panel close handlers ─────────────────────────────────────────────────

$('#panel-close').addEventListener('click', closeDetailPanel);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDetailPanel(); });

// ── Action handlers ──────────────────────────────────────────────────────

function showOutput(el, text) {
  el.textContent = text;
  el.classList.add('visible');
}

$('#assign-job-btn').addEventListener('click', async () => {
  const jobId = $('#assign-job-select').value;
  if (!state.selectedExperiment || !jobId) return;
  const data = await api('POST', '/api/assign-job', {
    experiment_name: state.selectedExperiment, job_id: jobId,
  });
  showOutput(actionOutput, data.message);
  await loadExperiments();
});

$('#cancel-job-btn').addEventListener('click', async () => {
  const exp = getSelectedExp();
  if (!exp || !exp.slurm_job || exp.slurm_job === NO_JOB) {
    showOutput(actionOutput, 'No SLURM Job ID assigned.');
    return;
  }
  const data = await api('POST', '/api/cancel-job', { job_id: exp.slurm_job });
  showOutput(actionOutput, data.message);
  await loadExperiments();
});

$('#eval-submit-btn').addEventListener('click', () => submitEval(false));
$('#eval-dryrun-btn').addEventListener('click', () => submitEval(true));

async function submitEval(dryRun) {
  if (!state.selectedExperiment) return;
  const dataset = $('#eval-dataset').value;
  const taskType = inferTaskType(dataset);
  const data = await api('POST', '/api/eval/submit', {
    experiment_name: state.selectedExperiment,
    dataset: dataset,
    n_rollouts: parseInt($('#eval-nrollouts').value),
    slurm_config: $('#eval-slurm-config').value,
    cpus_per_gpu: parseInt($('#eval-cpus-per-gpu').value),
    mem_per_gpu: $('#eval-mem-per-gpu').value,
    dry_run: dryRun,
    task_type: taskType,
    trajectory_analysis: $('#eval-trajectory').checked,
  });
  showOutput(actionOutput, data.output);
  if (!dryRun) await loadExperiments();
}

// ── Eval dataset population ──────────────────────────────────────────────

let evalDatasetsPopulated = false;

function populateEvalDatasets() {
  if (evalDatasetsPopulated) return;
  const evalDatasets = state.metadata.eval_datasets;
  if (!evalDatasets) return;

  const select = $('#eval-dataset');
  select.innerHTML = '';
  for (const [category, datasets] of Object.entries(evalDatasets)) {
    const group = document.createElement('optgroup');
    group.label = category;
    for (const ds of datasets) {
      const opt = document.createElement('option');
      opt.value = ds;
      opt.textContent = ds;
      group.appendChild(opt);
    }
    select.appendChild(group);
  }
  evalDatasetsPopulated = true;
}

// ── Launch form ──────────────────────────────────────────────────────────

async function loadSlurmConfigs() {
  try {
    const data = await api('GET', '/api/slurm-configs');
    const configs = data.configs || [];
    // Populate launch, eval, and resume SLURM config dropdowns
    for (const sel of [$('#launch-node'), $('#eval-slurm-config'), $('#resume-slurm-config')]) {
      if (!sel) continue;
      sel.innerHTML = '';
      for (const c of configs) {
        const opt = document.createElement('option');
        opt.value = c.name;
        opt.textContent = `${c.name} (${c.gpu_type || 'unknown'})`;
        sel.appendChild(opt);
      }
    }
  } catch (err) {
    console.error('loadSlurmConfigs error:', err);
  }
}

$('#launch-btn').addEventListener('click', () => doLaunch(false));
$('#launch-dryrun-btn').addEventListener('click', () => doLaunch(true));

async function doLaunch(dryRun) {
  const data = await api('POST', '/api/launch', {
    workflow: $('#launch-workflow').value,
    model: $('#launch-model').value,
    share_policy: $('#launch-policy').value,
    node: $('#launch-node').value,
    extra_args: $('#launch-extra').value,
    dry_run: dryRun,
    task_type: $('#launch-task-type').value,
    n_gpus: parseInt($('#launch-n-gpus').value),
    cpus_per_gpu: parseInt($('#launch-cpus-per-gpu').value),
    mem_per_gpu: $('#launch-mem-per-gpu').value,
  });
  showOutput(launchOutput, data.output);
  if (!dryRun) await loadExperiments();
}

// ── Resume handlers ──────────────────────────────────────────────────────

$('#resume-btn').addEventListener('click', () => doResume(false));
$('#resume-dryrun-btn').addEventListener('click', () => doResume(true));

async function doResume(dryRun) {
  const exp = getSelectedExp();
  if (!exp) return;

  // Auto-derive from experiment metadata
  const sharePolicy = exp.policy === 'share_policy' ? 'true' : 'false';
  const taskType = inferTaskType(exp.dataset);
  const nGpus = exp.gpu_count != null ? exp.gpu_count : parseInt($('#resume-n-gpus').value);

  const data = await api('POST', '/api/launch', {
    workflow: exp.workflow,
    model: exp.model,
    share_policy: sharePolicy,
    node: $('#resume-slurm-config').value,
    task_type: taskType,
    n_gpus: nGpus,
    cpus_per_gpu: parseInt($('#resume-cpus-per-gpu').value),
    mem_per_gpu: $('#resume-mem-per-gpu').value,
    extra_args: '',
    dry_run: dryRun,
  });
  showOutput(actionOutput, data.output);
  if (!dryRun) await loadExperiments();
}

// ── Refresh button ───────────────────────────────────────────────────────

$('#refresh-btn').addEventListener('click', loadExperiments);

// ── Auto-refresh ─────────────────────────────────────────────────────────

function startAutoRefresh() {
  stopAutoRefresh();
  state.refreshTimer = setInterval(loadExperiments, 30000);
}

function stopAutoRefresh() {
  if (state.refreshTimer) {
    clearInterval(state.refreshTimer);
    state.refreshTimer = null;
  }
}

$('#auto-refresh').addEventListener('change', (e) => {
  if (e.target.checked) startAutoRefresh();
  else stopAutoRefresh();
});

// ── Init ─────────────────────────────────────────────────────────────────

(async function init() {
  await Promise.all([loadExperiments(), loadSlurmConfigs()]);
  if ($('#auto-refresh').checked) startAutoRefresh();
})();
