// Cross-browser API
const api = (typeof browser !== 'undefined') ? browser : chrome;

const input    = document.getElementById('commandInput');
const outputEl = document.getElementById('output');
const runBtn   = document.getElementById('executeBtn');
const exitBtn  = document.getElementById('exitBtn');
const clearBtn = document.getElementById('clearBtn');

const HISTORY_KEY = 'qc_history'; // [{ cwd, cmd, msg, ok, ts }, ...] (max 10)

// --- Render helpers ---
function fmt(entry) {
  const status = entry.ok ? '✔' : '✖';
  const cwd    = entry.cwd ? entry.cwd : '';
  return `${cwd} $ ${entry.cmd}\n${status} ${entry.msg}`;
}

function renderAll(list) {
  outputEl.textContent = list.map(fmt).join('\n\n');
  outputEl.scrollTop = outputEl.scrollHeight;
}

function appendOne(entry) {
  const sep = outputEl.textContent ? '\n\n' : '';
  outputEl.textContent = outputEl.textContent + sep + fmt(entry);
  outputEl.scrollTop = outputEl.scrollHeight;
}

// --- Storage helpers ---
function loadHistory(cb) {
  api.storage.local.get([HISTORY_KEY], res => {
    cb(res[HISTORY_KEY] || []);
  });
}

function saveHistory(list, cb) {
  api.storage.local.set({ [HISTORY_KEY]: list }, cb || (() => {}));
}

function pushHistory(entry) {
  loadHistory(hist => {
    hist.push(entry);
    if (hist.length > 10) hist = hist.slice(-10);
    saveHistory(hist);
  });
}

// --- Command execution ---
function runCommand() {
  const cmd = (input.value || '').trim();
  if (!cmd) return;

  input.value = '';
  input.focus();

  api.runtime.sendMessage({ type: 'runCommand', command: cmd }, resp => {
    const ok  = resp && typeof resp.ok === 'boolean' ? resp.ok : true;
    const msg = resp && resp.msg ? resp.msg :
                (resp && resp.output ? resp.output : 'No response');
    const cwd = resp && resp.cwd ? resp.cwd : '';  // 👈 pick cwd from host

    const entry = { cwd, cmd, msg, ok, ts: Date.now() };
    appendOne(entry);
    pushHistory(entry);
  });
}

// --- Wire up UI ---
runBtn.addEventListener('click', runCommand);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter') runCommand();
});
exitBtn.addEventListener('click', () => window.close());
clearBtn.addEventListener('click', () => {
  saveHistory([], () => renderAll([]));
  input.focus();
});

// --- Load on open ---
document.addEventListener('DOMContentLoaded', () => {
  loadHistory(renderAll);
  input.focus();
});
