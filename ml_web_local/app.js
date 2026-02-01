import { trainPipeline, crc32 } from "./train.js";

const stepsEl = document.getElementById("steps");
const logEl = document.getElementById("log");
const spinnerEl = document.getElementById("spinner");
const connEl = document.getElementById("conn");
const btnReady = document.getElementById("btnReady");
const btnDisconnect = document.getElementById("btnDisconnect");
const btnDownloadTrace = document.getElementById("btnDownloadTrace");
const btnDownloadLog = document.getElementById("btnDownloadLog");
const resultsEl = document.getElementById("results");
const summaryEl = document.getElementById("summary");
const shotGridEl = document.getElementById("shotGrid");
const btnLoadModel = document.getElementById("btnLoadModel");
const btnDownloadModels = document.getElementById("btnDownloadModels");
const afterEl = document.getElementById("after");
const afterTextEl = document.getElementById("afterText");
const btnDone = document.getElementById("btnDone");
const btnResetFactory = document.getElementById("btnResetFactory");

const STEPS = [
  "Connect to blaster",
  "Pull log from device (MLDUMP)",
  "Train LR + MLP (in browser)",
  "Generate plots",
  "Ready to load models",
];

// Safety caps: prevents accidental multi-megabyte reads if the serial stream desyncs.
// LittleFS is ~1.5MB, so anything above ~2MB is definitely wrong.
const MAX_LOG_BYTES = 2 * 1024 * 1024;

let port = null;
let writer = null;
let serialIO = null;
let trainResult = null;
let rawLogBytes = null;

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

const TRACE = [];
const TRACE_T0 = performance.now();
function trace(kind, msg) {
  TRACE.push({ t_ms: Math.round(performance.now() - TRACE_T0), kind, msg });
  btnDownloadTrace.disabled = TRACE.length === 0;
}

function setSpinner(on) { spinnerEl.classList.toggle("hidden", !on); }

function log(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
  trace("log", msg);
}

function renderSteps(activeIdx, statusByIdx = {}) {
  stepsEl.replaceChildren();
  STEPS.forEach((s, i) => {
    const row = document.createElement("div"); row.className = "step";
    const left = document.createElement("div"); left.textContent = s;
    const tag = document.createElement("div"); tag.className = "tag";
    const st = statusByIdx[i] ?? (i < activeIdx ? "ok" : i === activeIdx ? "run" : "todo");
    if (st === "ok") { tag.textContent = "done"; tag.classList.add("ok"); }
    else if (st === "run") { tag.textContent = "running"; tag.classList.add("run"); }
    else if (st === "err") { tag.textContent = "error"; tag.classList.add("err"); }
    else { tag.textContent = "pending"; }
    row.appendChild(left); row.appendChild(tag); stepsEl.appendChild(row);
  });
}

// ── Serial IO ───────────────────────────────────────────────────────────────

async function ensureSerial() {
  if (!("serial" in navigator)) throw new Error("Web Serial not supported. Use Chrome/Edge.");
}

async function connect() {
  await ensureSerial();
  port = await navigator.serial.requestPort();
  await port.open({ baudRate: 115200 });
  try { await port.setSignals({ dataTerminalReady: true, requestToSend: true }); } catch {}
  writer = port.writable.getWriter();
  serialIO = new SerialIO(port);
  serialIO.clear();
  trace("serial", "port opened + DTR/RTS asserted");
  btnDisconnect.disabled = false; btnResetFactory.disabled = false;
  connEl.textContent = "Connected";
  log("Connected to serial.");
}

async function disconnect() {
  try {
    try { if (port) await port.setSignals({ dataTerminalReady: false, requestToSend: false }); } catch {}
    trace("serial", "DTR/RTS deasserted");
    if (serialIO) await serialIO.close();
    if (writer) { try { await writer.close(); } catch {} writer.releaseLock(); }
    if (port) await port.close();
  } finally {
    port = null; writer = null; serialIO = null;
    btnDisconnect.disabled = true; btnResetFactory.disabled = true; connEl.textContent = "Not connected";
    trace("serial", "port closed");
  }
}

async function writeText(s) {
  await writer.write(new TextEncoder().encode(s));
  trace("tx", s.replace(/\r/g, "\\r").replace(/\n/g, "\\n"));
}

async function readLine(timeoutMs = 5000) { return serialIO.readLine(timeoutMs); }

async function resyncSerial(label = "Resyncing serial session...") {
  log(label); serialIO.clear();
  try { await writeText("\n"); } catch {}
  await sleep(60);
  trace("serial", "resync: cleared buffers");
}

async function hardResyncSerial(label = "Resyncing serial session...") {
  log(label);
  if (port) {
    try { await port.setSignals({ dataTerminalReady: false, requestToSend: false }); } catch {}
    await sleep(120);
    try { await port.setSignals({ dataTerminalReady: true, requestToSend: true }); } catch {}
    await sleep(120);
  }
  serialIO.clear();
  trace("serial", "resync: toggled DTR/RTS + cleared buffers");
}

async function drainSerial(ms) {
  const deadline = performance.now() + ms;
  while (performance.now() < deadline) {
    try { const line = await readLine(120); log("<< " + line); }
    catch (e) { if (!(e?.message ?? "").includes("Timeout")) throw e; await sleep(20); }
  }
}

// ── Pull log ────────────────────────────────────────────────────────────────

async function pullLog() {
  async function attemptOnce() {
    await writeText("MLDUMP\n"); log("Sent MLDUMP");
    let size = null;
    for (let i = 0; i < 40; i++) {
      const line = await readLine(4000); log("<< " + line);
      if (line.startsWith("MLDUMP1 ")) {
        const n = parseInt(line.slice("MLDUMP1 ".length).trim(), 10);
        if (Number.isFinite(n) && n > 0) size = n; break;
      }
    }
    if (!size || size <= 0) throw new Error("Did not receive MLDUMP1 <size> header");
    if (size > MAX_LOG_BYTES) throw new Error(`Refusing to read ${size} bytes (cap ${MAX_LOG_BYTES}).`);
    log(`Expecting ${size} bytes...`);
    if (size % 17 !== 0) log(`WARNING: log size ${size} not a multiple of 17 bytes/sample.`);
    const progressT0 = performance.now();
    let lastUi = 0, lastBytes = 0;
    const blob = await serialIO.readExactWithProgress(size, 180000, (got) => {
      const now = performance.now();
      if (now - lastUi < 250) return; lastUi = now;
      const dt = (now - progressT0) / 1000;
      const kbps = dt > 0 ? (got / 1024) / dt : 0; lastBytes = got;
      log(`MLDUMP progress: ${got}/${size} bytes (${kbps.toFixed(1)} KB/s)`);
      trace("mldump", `${got}/${size} bytes`);
    });
    log(`Read ${blob.length} bytes`);
    const doneDeadline = performance.now() + 1500;
    while (performance.now() < doneDeadline) {
      try { const line = await readLine(250);
        if (line.includes("MLDUMP_DONE")) { log("<< MLDUMP_DONE"); break; }
      } catch (e) { if (!(e?.message ?? "").includes("Timeout")) throw e; }
    }
    return blob;
  }
  await resyncSerial("Resyncing serial session...");
  try { return await attemptOnce(); }
  catch (e) { log("MLDUMP failed, retrying once...");
    await hardResyncSerial("Resyncing and retrying MLDUMP..."); return await attemptOnce(); }
}

// ── Model upload ────────────────────────────────────────────────────────────

async function uploadModel(cmdPrefix, u8) {
  await drainSerial(300);
  if (u8.length <= 0 || u8.length > 200000) throw new Error(`Invalid model size: ${u8.length} bytes`);
  const c = crc32(u8);
  await writeText(`${cmdPrefix} ${u8.length} ${c.toString(16).padStart(8, "0")}\n`);
  log(`Sent ${cmdPrefix} (${u8.length} bytes, crc=${c.toString(16)})`);
  let ready = false;
  const readyDeadline = performance.now() + 15000;
  while (performance.now() < readyDeadline) {
    try { const line = await readLine(800); log("<< " + line);
      if (line.includes("MLMODEL_READY")) { ready = true; break; }
      if (line.includes("MLMODEL_ERR")) throw new Error(line);
    } catch (e) { if ((e?.message ?? "").includes("Timeout")) continue; throw e; }
  }
  if (!ready) throw new Error("No MLMODEL_READY received.");
  const chunk = 512;
  for (let i = 0; i < u8.length; i += chunk) await writer.write(u8.subarray(i, i + chunk));
  log("Uploaded bytes, waiting for OK...");
  const okDeadline = performance.now() + 15000;
  while (performance.now() < okDeadline) {
    try { const line = await readLine(800); log("<< " + line);
      if (line.includes("MLMODEL_OK")) return;
      if (line.includes("MLMODEL_ERR")) throw new Error(line);
    } catch (e) { if ((e?.message ?? "").includes("Timeout")) continue; throw e; }
  }
  throw new Error("Timeout waiting for MLMODEL_OK");
}

async function loadModel(cmd) {
  await drainSerial(250);
  await writeText(cmd + "\n"); log("Sent " + cmd);
  const deadline = performance.now() + 8000;
  while (performance.now() < deadline) {
    try { const line = await readLine(800); log("<< " + line);
      if (line.includes("MLMODEL_LOADED")) return;
      if (line.includes("MLMODEL_ERR")) throw new Error(line);
    } catch (e) { if ((e?.message ?? "").includes("Timeout")) continue; throw e; }
  }
  throw new Error("Timeout waiting for MLMODEL_LOADED");
}

async function queryModelInfo() {
  await drainSerial(250);
  await writeText("MLMODEL_INFO\n"); log("Sent MLMODEL_INFO");
  const deadline = performance.now() + 4000;
  let lastRx = performance.now(), gotMarker = false;
  while (performance.now() < deadline) {
    try { const line = await readLine(500); lastRx = performance.now(); log("<< " + line);
      if (line.includes("model_source=") || line.includes("user_has_lr=")) gotMarker = true;
    } catch (e) { if (!(e?.message ?? "").includes("Timeout")) throw e;
      if (gotMarker && performance.now() - lastRx > 250) break; }
  }
  if (!gotMarker) throw new Error("No MLMODEL_INFO response.");
}

// ── Canvas plots ────────────────────────────────────────────────────────────

function drawShotPlot(canvas, shot) {
  const dpr = window.devicePixelRatio || 1;
  const W = 540, H = 380;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = "100%"; canvas.style.height = "auto";
  canvas.style.aspectRatio = W + "/" + H;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const pad = { l: 44, r: 16, t: 28, mid: 18, b: 22 };
  const topH = 120, botH = 150;
  const plotW = W - pad.l - pad.r;

  const xMs = shot.xMs;
  const xMin = xMs[0], xMax = xMs[xMs.length - 1];
  const xScale = (x) => pad.l + (x - xMin) / (xMax - xMin) * plotW;

  ctx.fillStyle = "#0b0e14"; ctx.fillRect(0, 0, W, H);

  const predL = xScale(-600), predR = xScale(-100);
  const trigX = xScale(0);

  // ── Section title: predictions ──
  ctx.fillStyle = "#9aa4b2"; ctx.font = "bold 10px system-ui, sans-serif"; ctx.textAlign = "left";
  ctx.fillText("p(shoot)", pad.l, pad.t - 10);

  // Legend top-right
  const legY = pad.t - 16;
  ctx.fillStyle = "#38bdf8"; ctx.fillRect(W - pad.r - 68, legY, 10, 3);
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillStyle = "#9aa4b2"; ctx.fillText("LR", W - pad.r - 54, legY + 4);
  ctx.fillStyle = "#22c55e"; ctx.fillRect(W - pad.r - 30, legY, 10, 3);
  ctx.fillStyle = "#9aa4b2"; ctx.fillText("MLP", W - pad.r - 16, legY + 4);

  // ── Top: predictions plot ──
  const topY0 = pad.t, topY1 = pad.t + topH;
  ctx.fillStyle = "rgba(255,255,255,0.04)";
  ctx.fillRect(predL, topY0, predR - predL, topH);

  ctx.strokeStyle = "rgba(255,60,60,0.8)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(trigX, topY0); ctx.lineTo(trigX, topY1); ctx.stroke();

  // Y-axis labels
  ctx.fillStyle = "#9aa4b2"; ctx.font = "9px system-ui, sans-serif"; ctx.textAlign = "right";
  ctx.fillText("1.0", pad.l - 5, topY0 + 8);
  ctx.fillText("0.5", pad.l - 5, (topY0 + topY1) / 2 + 3);
  ctx.fillText("0.0", pad.l - 5, topY1 + 1);

  // Grid lines
  const probYScale = (p) => topY1 - p * topH;
  ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.lineWidth = 0.5;
  for (const p of [0.25, 0.5, 0.75]) {
    const y = probYScale(p);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
  }

  // Probability lines
  function drawLine(data, color) {
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = xScale(xMs[i]), y = probYScale(data[i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  drawLine(shot.probsLR, "#38bdf8");
  drawLine(shot.probsMLP, "#22c55e");

  // ── Section title: IMU ──
  const botY0 = topY1 + pad.mid, botY1 = botY0 + botH;
  ctx.fillStyle = "#9aa4b2"; ctx.font = "bold 10px system-ui, sans-serif"; ctx.textAlign = "left";
  ctx.fillText("IMU (raw i16)", pad.l, botY0 - 6);

  // IMU legend top-right
  const imuColors = ["#60a5fa", "#4ade80", "#fb923c", "#f87171", "#c084fc", "#a1887f"];
  const imuLabels = ["ax", "ay", "az", "gx", "gy", "gz"];
  ctx.font = "9px system-ui, sans-serif";
  let legX = W - pad.r;
  for (let ch = 5; ch >= 0; ch--) {
    const tw = ctx.measureText(imuLabels[ch]).width;
    legX -= tw;
    ctx.fillStyle = imuColors[ch]; ctx.fillText(imuLabels[ch], legX, botY0 - 6);
    legX -= 12;
  }

  // ── Bottom: IMU plot ──
  ctx.fillStyle = "rgba(255,255,255,0.04)";
  ctx.fillRect(predL, botY0, predR - predL, botH);

  ctx.strokeStyle = "rgba(255,60,60,0.8)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(trigX, botY0); ctx.lineTo(trigX, botY1); ctx.stroke();

  let imuMin = Infinity, imuMax = -Infinity;
  for (const arr of [shot.axW, shot.ayW, shot.azW, shot.gxW, shot.gyW, shot.gzW])
    for (const v of arr) { if (v < imuMin) imuMin = v; if (v > imuMax) imuMax = v; }
  const imuPad = (imuMax - imuMin) * 0.05 + 1;
  imuMin -= imuPad; imuMax += imuPad;
  const imuYScale = (v) => botY1 - (v - imuMin) / (imuMax - imuMin) * botH;

  const imuData = [shot.axW, shot.ayW, shot.azW, shot.gxW, shot.gyW, shot.gzW];
  for (let ch = 0; ch < 6; ch++) {
    ctx.strokeStyle = imuColors[ch]; ctx.lineWidth = 0.8; ctx.globalAlpha = 0.7; ctx.beginPath();
    for (let i = 0; i < imuData[ch].length; i++) {
      const x = xScale(xMs[i]), y = imuYScale(imuData[ch][i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // ── Shared X-axis at bottom ──
  ctx.fillStyle = "#9aa4b2"; ctx.font = "9px system-ui, sans-serif"; ctx.textAlign = "center";
  const step = 200;
  for (let ms = Math.ceil(xMin / step) * step; ms <= xMax; ms += step) {
    const x = xScale(ms);
    ctx.fillText(ms + "ms", x, botY1 + 14);
  }
}

// ── Results display ─────────────────────────────────────────────────────────

function showResults(result) {
  resultsEl.classList.remove("hidden");
  const s = result.summary;
  summaryEl.replaceChildren();
  const rows = [
    ["Duration:", `${(s.durationMs / 1000).toFixed(1)}s (${s.samples} samples)`],
    ["Shots:", `${s.shotsAll} total, ${s.shotsAccepted} accepted`],
    ["Training windows:", `${s.posWindows} positive, ${s.negWindows} negative`],
  ];
  for (const [k, v] of rows) {
    const div = document.createElement("div");
    const b = document.createElement("b"); b.textContent = k;
    div.appendChild(b);
    div.appendChild(document.createTextNode(" " + v));
    summaryEl.appendChild(div);
  }

  shotGridEl.replaceChildren();
  if (result.shotPlots.length === 0) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.textContent = "No shots had strong enough predictions to display. The model may need more training data.";
    shotGridEl.appendChild(empty);
  }
  for (const shot of result.shotPlots) {
    const div = document.createElement("div"); div.className = "shot";
    const canvas = document.createElement("canvas");
    div.appendChild(canvas); shotGridEl.appendChild(div);
    drawShotPlot(canvas, shot);
  }
}

// ── Download helpers ────────────────────────────────────────────────────────

function downloadBlob(name, u8) {
  const blob = new Blob([u8], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}

function downloadText(name, text) {
  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}

// ── Main flow ───────────────────────────────────────────────────────────────

btnReady.addEventListener("click", async () => {
  logEl.textContent = "";
  resultsEl.classList.add("hidden"); afterEl.classList.add("hidden");
  btnLoadModel.disabled = true; btnDownloadModels.disabled = true; btnDownloadLog.disabled = true;
  trainResult = null; rawLogBytes = null; btnReady.disabled = true;

  const status = {};
  try {
    renderSteps(0); setSpinner(true);

    status[0] = "run"; renderSteps(0, status);
    await connect(); status[0] = "ok";

    status[1] = "run"; renderSteps(1, status);
    const logBytes = await pullLog(); rawLogBytes = logBytes; status[1] = "ok";

    status[2] = "run"; renderSteps(2, status);
    await sleep(50); // yield to UI
    const result = trainPipeline(logBytes, log);
    trainResult = result; status[2] = "ok";

    status[3] = "ok"; status[4] = "ok";
    renderSteps(4, status); setSpinner(false);
    log("Training complete.");

    showResults(result);
    btnLoadModel.disabled = false; btnDownloadModels.disabled = false;
    btnDownloadLog.disabled = false; btnReady.disabled = false;
  } catch (e) {
    setSpinner(false); log("ERROR: " + (e?.message ?? String(e)));
    const idx = Object.keys(status).length ? Math.max(...Object.keys(status).map((x) => parseInt(x, 10))) : 0;
    status[idx] = "err"; renderSteps(idx, status); btnReady.disabled = false;
  }
});

btnDisconnect.addEventListener("click", async () => { await disconnect(); });

btnDownloadTrace.addEventListener("click", () => {
  const payload = { meta: { userAgent: navigator.userAgent, time: new Date().toISOString() }, trace: TRACE };
  downloadText(`stinger_ml_web_trace_${Date.now()}.json`, JSON.stringify(payload, null, 2));
});

btnDownloadModels.addEventListener("click", () => {
  if (!trainResult) return;
  downloadBlob("ml_model_lr.bin", trainResult.lrBlob);
  downloadBlob("ml_model_mlp.bin", trainResult.mlpBlob);
});

btnDownloadLog.addEventListener("click", () => {
  if (!rawLogBytes) return;
  downloadBlob("ml_log.bin", rawLogBytes);
});

btnLoadModel.addEventListener("click", async () => {
  if (!trainResult) return;
  try {
    btnLoadModel.disabled = true; setSpinner(true);
    await queryModelInfo();
    log("Uploading LR..."); await uploadModel("MLMODEL_PUT_LR", trainResult.lrBlob);
    log("Uploading MLP..."); await uploadModel("MLMODEL_PUT_MLP", trainResult.mlpBlob);
    log("Loading LR..."); await loadModel("MLMODEL_LOAD_LR");
    log("Loading MLP..."); await loadModel("MLMODEL_LOAD_MLP");
    log("Verifying loaded weights..."); await queryModelInfo();
    setSpinner(false); afterEl.classList.remove("hidden");
    afterTextEl.textContent = "Models loaded successfully! You can now disconnect and try it out. Compare ML:LR vs ML:MLP in Motor \u2192 Idling.";
  } catch (e) { setSpinner(false); btnLoadModel.disabled = false;
    log("ERROR while uploading: " + (e?.message ?? String(e))); }
});

btnDone.addEventListener("click", async () => { await disconnect(); });

btnResetFactory.addEventListener("click", async () => {
  if (!serialIO) return;
  if (!confirm("This will delete your personal model and revert to factory defaults. Continue?")) return;
  try {
    btnResetFactory.disabled = true;
    await drainSerial(300);
    await writeText("MLMODEL_DELETE\n");
    log("Sent MLMODEL_DELETE");
    let deleted = false;
    for (let i = 0; i < 20; i++) {
      const line = await readLine(2000); log("<< " + line);
      if (line.includes("MLMODEL_DELETED")) { deleted = true; break; }
    }
    if (deleted) {
      log("Factory model restored.");
      await queryModelInfo();
      alert("Factory model restored. You can disconnect now.");
    } else {
      log("WARNING: Did not receive MLMODEL_DELETED confirmation.");
    }
  } catch (e) { log("ERROR: " + (e?.message ?? String(e))); }
  finally { btnResetFactory.disabled = false; }
});

// ── SerialIO class ──────────────────────────────────────────────────────────

class SerialIO {
  constructor(port) {
    this.port = port; this.reader = port.readable.getReader();
    this.chunks = []; this.headOff = 0; this.total = 0;
    this._waiters = []; this._closed = false; this._pumpErr = null;
    this._pumpPromise = this._pump();
  }
  _remaining() { return this.total; }
  _take(n) {
    const out = new Uint8Array(n); let outOff = 0;
    while (outOff < n) {
      const head = this.chunks[0]; const avail = head.length - this.headOff;
      const take = Math.min(avail, n - outOff);
      out.set(head.subarray(this.headOff, this.headOff + take), outOff);
      outOff += take; this.headOff += take; this.total -= take;
      if (this.headOff >= head.length) { this.chunks.shift(); this.headOff = 0; }
    }
    return out;
  }
  _append(chunk) { if (!chunk?.length) return; this.chunks.push(chunk); this.total += chunk.length; this._notify(); }
  _notify() { const w = this._waiters; this._waiters = []; w.forEach((fn) => fn()); }
  async _pump() {
    try { while (true) { const { value, done } = await this.reader.read(); if (done) break; if (value?.length) this._append(value); } }
    catch (e) { this._pumpErr = e; } finally { this._closed = true; this._notify(); }
  }
  async close() {
    try { if (this.reader) { try { await this.reader.cancel(); } catch {} this.reader.releaseLock(); } }
    finally { this._closed = true; this._notify(); }
  }
  _closedError() { return this._pumpErr || new Error("Serial reader closed"); }
  async _waitForData(timeoutMs) {
    if (this._remaining() > 0) return; if (this._closed) throw this._closedError();
    await Promise.race([
      new Promise((r) => this._waiters.push(r)),
      sleep(timeoutMs).then(() => { throw new Error("Timeout waiting for serial data"); }),
    ]);
    if (this._closed) throw this._closedError();
  }
  _findNewlinePos() {
    let pos = 0;
    for (let ci = 0; ci < this.chunks.length; ci++) {
      const chunk = this.chunks[ci]; const start = ci === 0 ? this.headOff : 0;
      for (let i = start; i < chunk.length; i++) if (chunk[i] === 0x0a) return pos + (i - start) + 1;
      pos += chunk.length - start;
    }
    return -1;
  }
  async readLine(timeoutMs = 5000) {
    const deadline = performance.now() + timeoutMs;
    while (true) {
      while (true) { const want = this._findNewlinePos(); if (want < 0) break;
        const lineBytes = this._take(want);
        let s = new TextDecoder("utf-8", { fatal: false }).decode(lineBytes).replace(/\r/g, "").trim();
        if (s.length) return s;
      }
      const left = Math.max(0, Math.floor(deadline - performance.now()));
      if (left <= 0) throw new Error("Timeout waiting for line");
      await this._waitForData(left);
    }
  }
  async readExact(n, timeoutMs = 15000) {
    const deadline = performance.now() + timeoutMs; const out = new Uint8Array(n); let off = 0;
    while (off < n) {
      const avail = this._remaining();
      if (avail > 0) { const take = Math.min(avail, n - off); out.set(this._take(take), off); off += take; continue; }
      const left = Math.max(0, Math.floor(deadline - performance.now())); if (left <= 0) break;
      await this._waitForData(left);
    }
    if (off !== n) throw new Error(`Timeout reading binary (${off}/${n})`); return out;
  }
  async readExactWithProgress(n, timeoutMs, onProgress) {
    const deadline = performance.now() + timeoutMs; const out = new Uint8Array(n); let off = 0;
    while (off < n) {
      const avail = this._remaining();
      if (avail > 0) { const take = Math.min(avail, n - off); out.set(this._take(take), off); off += take;
        try { onProgress?.(off); } catch {} continue; }
      const left = Math.max(0, Math.floor(deadline - performance.now())); if (left <= 0) break;
      await this._waitForData(left);
    }
    if (off !== n) throw new Error(`Timeout reading binary (${off}/${n})`); return out;
  }
  clear() { this.chunks = []; this.headOff = 0; this.total = 0; }
}
