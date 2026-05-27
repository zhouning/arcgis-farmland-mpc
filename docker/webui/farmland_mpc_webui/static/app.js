const form = document.getElementById('run-form');
const statusCard = document.getElementById('status-card');
const jobIdEl = document.getElementById('job-id');
const phaseEl = document.getElementById('phase');
const elapsedEl = document.getElementById('elapsed');
const logEl = document.getElementById('log');
const barFill = document.getElementById('bar-fill');
const downloadEl = document.getElementById('download');
const errorEl = document.getElementById('error');

let pollHandle = null;

form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const fd = new FormData(form);
  errorEl.classList.add('hidden');
  downloadEl.classList.add('hidden');
  logEl.textContent = '';
  barFill.style.width = '0%';
  phaseEl.textContent = 'uploading…';
  statusCard.classList.remove('hidden');

  const r = await fetch('/run', { method: 'POST', body: fd });
  if (!r.ok) {
    errorEl.textContent = 'upload failed: ' + (await r.text());
    errorEl.classList.remove('hidden');
    return;
  }
  const { job_id } = await r.json();
  jobIdEl.textContent = `· ${job_id}`;
  if (pollHandle) clearInterval(pollHandle);
  pollHandle = setInterval(() => poll(job_id), 2000);
  poll(job_id);
});

async function poll(jobId) {
  const r = await fetch(`/status/${jobId}`);
  if (!r.ok) return;
  const s = await r.json();
  phaseEl.textContent = s.phase;
  elapsedEl.textContent = formatElapsed(s.elapsed_s);
  barFill.style.width = (s.progress * 100).toFixed(0) + '%';
  logEl.textContent = s.log_tail || '';
  logEl.scrollTop = logEl.scrollHeight;

  if (s.phase === 'done') {
    clearInterval(pollHandle);
    downloadEl.href = `/download/${jobId}`;
    downloadEl.classList.remove('hidden');
  } else if (s.phase === 'error') {
    clearInterval(pollHandle);
    errorEl.textContent = s.error || 'failed (see log above)';
    errorEl.classList.remove('hidden');
  }
}

function formatElapsed(s) {
  s = Math.round(s);
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
