import { trainPipeline, crc32 } from "./train.js";

const stepsEl = document.getElementById("steps");
const logEl = document.getElementById("log");
const spinnerEl = document.getElementById("spinner");
const nnVizEl = document.getElementById("nnViz");
const nnVizCanvasEl = document.getElementById("nnVizCanvas");
const nnVizSubtitleEl = document.getElementById("nnVizSubtitle");
const connEl = document.getElementById("conn");
const btnReady = document.getElementById("btnReady");
const btnDisconnect = document.getElementById("btnDisconnect");
const btnDownloadTrace = document.getElementById("btnDownloadTrace");
const btnDownloadLog = document.getElementById("btnDownloadLog");
const resultsEl = document.getElementById("results");
const summaryEl = document.getElementById("summary");
const metricsEl = document.getElementById("metrics");
const shotGridEl = document.getElementById("shotGrid");
const btnLoadModel = document.getElementById("btnLoadModel");
const btnDownloadModels = document.getElementById("btnDownloadModels");
const afterEl = document.getElementById("after");
const afterTextEl = document.getElementById("afterText");
const btnDone = document.getElementById("btnDone");
const btnResetFactory = document.getElementById("btnResetFactory");
const dataQualityEl = document.getElementById("dataQuality");
const lossCanvasEl = document.getElementById("lossCanvas");
const btnPrintPlots = document.getElementById("btnPrintPlots");
const btnImportLogs = document.getElementById("btnImportLogs");
const fileInputEl = document.getElementById("fileInput");
const importStatusEl = document.getElementById("importStatus");

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

class MlpViz {
  constructor(canvasEl, subtitleEl) {
    this.canvas = canvasEl;
    this.subtitleEl = subtitleEl;
    this.active = false;
    this._raf = 0;
    this._t0 = 0;
    this._lastSnapshot = null;
    this._wScale = 1;
  }

  show() {
    if (this.active) return;
    this.active = true;
    this._t0 = performance.now();
    nnVizEl.classList.remove("hidden");
    this._resizeCanvas();
    this._raf = requestAnimationFrame(() => this._draw());
  }

  hide() {
    this.active = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = 0;
    nnVizEl.classList.add("hidden");
  }

  setSubtitle(text) { this.subtitleEl.textContent = text; }

  updateSnapshot(snap) {
    this._lastSnapshot = snap;
    if (!snap?.w1 || !snap?.w2 || !snap?.w3) return;
    const maxAbs = Math.max(maxAbsF32(snap.w1), maxAbsF32(snap.w2), maxAbsF32(snap.w3), 1e-6);
    this._wScale = 1 / maxAbs;
    this.setSubtitle(`Epoch ${snap.iter}/${snap.maxIter} · updating weights…`);
  }

  _resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const W = Math.min(980, nnVizEl.clientWidth - 40);
    const H = 420;
    this.canvas.width = Math.floor(W * dpr);
    this.canvas.height = Math.floor(H * dpr);
    this.canvas.style.width = W + "px";
    this.canvas.style.height = H + "px";
  }

  _draw() {
    if (!this.active) return;
    const dpr = window.devicePixelRatio || 1;
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const W = parseFloat(this.canvas.style.width);
    const H = parseFloat(this.canvas.style.height);
    ctx.clearRect(0, 0, W, H);

    // Background shimmer
    ctx.fillStyle = "rgba(0,0,0,0.22)";
    ctx.fillRect(0, 0, W, H);
    const t = (performance.now() - this._t0) / 1000;
    const g = ctx.createLinearGradient(0, 0, W, H);
    g.addColorStop(0, `rgba(56,189,248,${0.06 + 0.02 * Math.sin(t * 1.7)})`);
    g.addColorStop(0.5, `rgba(34,197,94,${0.05 + 0.02 * Math.cos(t * 1.3)})`);
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    this._drawNetwork(ctx, W, H);

    this._raf = requestAnimationFrame(() => this._draw());
  }

  _drawNetwork(ctx, W, H) {
    const pad = 22;
    const t = (performance.now() - this._t0) / 1000;
    const snap = this._lastSnapshot;

    // Real network: 30→64→32→1. We draw a readable subset.
    const nFeat = snap?.nFeat ?? 30;
    const h1Size = snap?.h1Size ?? 64;
    const h2Size = snap?.h2Size ?? 32;

    const inN = 8;
    const h1N = 12;
    const h2N = 8;

    const xIn = pad + 45;
    const xH1 = pad + (W - pad * 2) * 0.40;
    const xH2 = pad + (W - pad * 2) * 0.72;
    const xOut = W - pad - 55;

    const yTop = pad + 34;
    const yBot = H - pad - 34;
    const span = (yBot - yTop);
    const yFor = (i, n) => yTop + (n === 1 ? span / 2 : (span * i) / (n - 1));

    const pick = (nTotal, nShow) =>
      Array.from({ length: nShow }, (_, i) => Math.floor((i / Math.max(1, nShow - 1)) * (nTotal - 1)));

    const inIdx = pick(nFeat, inN);
    const h1Idx = pick(h1Size, h1N);
    const h2Idx = pick(h2Size, h2N);

    const inPos = inIdx.map((_, i) => ({ x: xIn, y: yFor(i, inN) }));
    const h1Pos = h1Idx.map((_, i) => ({ x: xH1, y: yFor(i, h1N) }));
    const h2Pos = h2Idx.map((_, i) => ({ x: xH2, y: yFor(i, h2N) }));
    const outPos = [{ x: xOut, y: yFor(0, 1) }];

    const wScale = this._wScale || 1;

    const edgeStyle = (w, phase) => {
      const s = Math.tanh(Math.abs(w) * wScale * 2.3); // [0,1)
      const a = 0.10 + 0.62 * s;
      const lw = 0.6 + 2.8 * s;
      const hue = w >= 0 ? 145 : 196; // green / blue
      const col = `hsla(${hue}, 85%, 60%, ${a})`;
      const dash = 10 + 18 * (1 - s) + 5 * Math.sin(phase);
      return { col, lw, glow: 0.18 + 0.62 * s, dash };
    };

    const drawEdge = (a, b, w, phase) => {
      const st = edgeStyle(w, phase);
      ctx.save();
      ctx.strokeStyle = st.col;
      ctx.lineWidth = st.lw;
      ctx.lineCap = "round";
      ctx.setLineDash([st.dash, st.dash * 0.85]);
      ctx.lineDashOffset = -t * (40 + 90 * st.glow);
      ctx.shadowColor = st.col;
      ctx.shadowBlur = 18 * st.glow;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      const cx = (a.x + b.x) / 2;
      ctx.bezierCurveTo(cx, a.y, cx, b.y, b.x, b.y);
      ctx.stroke();
      ctx.restore();
    };

    if (!snap?.w1 || !snap?.w2 || !snap?.w3) {
      ctx.fillStyle = "rgba(154,164,178,0.9)";
      ctx.font = "14px system-ui, sans-serif";
      ctx.fillText("Waiting for weights…", pad, pad + 18);
    } else {
      const w1 = snap.w1, w2 = snap.w2, w3 = snap.w3;

      for (let j = 0; j < h1N; j++) {
        const hj = h1Idx[j];
        for (let i = 0; i < inN; i++) {
          const fi = inIdx[i];
          drawEdge(inPos[i], h1Pos[j], w1[hj * nFeat + fi], i * 0.6 + j * 0.2);
        }
      }
      for (let k = 0; k < h2N; k++) {
        const hk = h2Idx[k];
        for (let j = 0; j < h1N; j++) {
          const hj = h1Idx[j];
          drawEdge(h1Pos[j], h2Pos[k], w2[hk * h1Size + hj], j * 0.35 + k * 0.3 + 1.1);
        }
      }
      for (let k = 0; k < h2N; k++) {
        const hk = h2Idx[k];
        drawEdge(h2Pos[k], outPos[0], w3[hk], k * 0.55 + 2.2);
      }
    }

    const drawNode = (p, r, fill, stroke, phase) => {
      const pulse = 1 + 0.07 * Math.sin(t * 2.1 + phase);
      ctx.save();
      ctx.beginPath();
      ctx.arc(p.x, p.y, r * pulse, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.shadowColor = stroke;
      ctx.shadowBlur = 16;
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = stroke;
      ctx.stroke();
      ctx.restore();
    };

    for (let i = 0; i < inPos.length; i++)
      drawNode(inPos[i], 7, "rgba(56,189,248,0.10)", "rgba(56,189,248,0.95)", i * 0.4);
    for (let j = 0; j < h1Pos.length; j++)
      drawNode(h1Pos[j], 7, "rgba(34,197,94,0.10)", "rgba(34,197,94,0.95)", j * 0.3 + 0.9);
    for (let k = 0; k < h2Pos.length; k++)
      drawNode(h2Pos[k], 8, "rgba(34,197,94,0.12)", "rgba(34,197,94,0.98)", k * 0.32 + 1.8);
    drawNode(outPos[0], 10, "rgba(56,189,248,0.14)", "rgba(56,189,248,0.98)", 2.8);

    ctx.fillStyle = "rgba(154,164,178,0.95)";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText("Inputs", xIn - 18, pad + 12);
    ctx.fillText("Hidden 1", xH1 - 26, pad + 12);
    ctx.fillText("Hidden 2", xH2 - 26, pad + 12);
    ctx.fillText("p(shoot)", xOut - 24, pad + 12);

    if (snap?.iter != null && snap?.maxIter) {
      const prog = clamp01(snap.iter / snap.maxIter);
      const barW = W - pad * 2;
      const barY = H - pad - 10;
      ctx.fillStyle = "rgba(255,255,255,0.06)";
      ctx.fillRect(pad, barY, barW, 6);
      const grad = ctx.createLinearGradient(pad, 0, pad + barW, 0);
      grad.addColorStop(0, "rgba(56,189,248,0.85)");
      grad.addColorStop(1, "rgba(34,197,94,0.85)");
      ctx.fillStyle = grad;
      ctx.fillRect(pad, barY, barW * prog, 6);
    }
  }
}

function maxAbsF32(arr) {
  let m = 0;
  for (let i = 0; i < arr.length; i++) {
    const a = Math.abs(arr[i]);
    if (a > m) m = a;
  }
  return m;
}

function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

function renderHeatmap(weights, rows, cols, scale) {
  const c = document.createElement("canvas");
  c.width = cols;
  c.height = rows;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(cols, rows);
  const data = img.data;
  for (let r = 0; r < rows; r++) {
    for (let col = 0; col < cols; col++) {
      const v = weights[r * cols + col] * scale;
      const t = Math.tanh(v * 2.2); // [-1,1] with some contrast
      const a = 0.18 + 0.82 * Math.abs(t);
      const idx = (r * cols + col) * 4;
      if (t >= 0) {
        // green
        data[idx] = 34;
        data[idx + 1] = 197;
        data[idx + 2] = 94;
      } else {
        // blue
        data[idx] = 56;
        data[idx + 1] = 189;
        data[idx + 2] = 248;
      }
      data[idx + 3] = Math.floor(255 * a);
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function drawHeatBlock(ctx, xLabel, yLabel, title, heatCanvas, x, y, w, h) {
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = "#9aa4b2";
  ctx.fillText(title, xLabel, yLabel + 4);

  ctx.fillStyle = "rgba(255,255,255,0.05)";
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = "rgba(255,255,255,0.10)";
  ctx.strokeRect(x, y, w, h);

  if (heatCanvas) {
    ctx.imageSmoothingEnabled = false;
    ctx.globalAlpha = 0.95;
    ctx.drawImage(heatCanvas, x + 1, y + 1, w - 2, h - 2);
    ctx.globalAlpha = 1;

    // subtle scanline shimmer
    const t = performance.now() / 1000;
    const scanY = y + ((Math.sin(t * 1.1) * 0.5 + 0.5) * (h - 10));
    const g = ctx.createLinearGradient(x, scanY, x, scanY + 18);
    g.addColorStop(0, "rgba(255,255,255,0)");
    g.addColorStop(0.5, "rgba(255,255,255,0.06)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(x + 1, scanY, w - 2, 18);
  } else {
    ctx.fillStyle = "rgba(154,164,178,0.85)";
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText("waiting for weights…", x + 12, y + 22);
  }
}

const mlpViz = new MlpViz(nnVizCanvasEl, nnVizSubtitleEl);

function setTrainingViz(on) {
  if (on) mlpViz.show();
  else mlpViz.hide();
}

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
  // Important: do NOT write a bare newline here.
  // Firmware intentionally only consumes serial bytes once it sees an 'M' prefix
  // (to avoid interfering with other Serial consumers). Sending '\n' can get stuck
  // at the head of the RX buffer and prevent all subsequent ML commands from being
  // parsed, leading to timeouts.
  log(label);
  serialIO.clear();
  await sleep(30);
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
      if (line.includes("[ml] ERROR:")) throw new Error(line);
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

function drawShotPlot(canvas, shot, print = false) {
  const dpr = print ? 3 : (window.devicePixelRatio || 1);
  const W = print ? 1080 : 540, H = print ? 760 : 380;
  canvas.width = W * dpr; canvas.height = H * dpr;
  if (!print) { canvas.style.width = "100%"; canvas.style.height = "auto"; canvas.style.aspectRatio = W + "/" + H; }
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  // Theme
  const bg = print ? "#ffffff" : "#0b0e14";
  const text = print ? "#333333" : "#9aa4b2";
  const grid = print ? "rgba(0,0,0,0.08)" : "rgba(255,255,255,0.06)";
  const band = print ? "rgba(0,0,0,0.04)" : "rgba(255,255,255,0.04)";
  const trig = print ? "rgba(220,40,40,0.7)" : "rgba(255,60,60,0.8)";
  const lrCol = print ? "#0077cc" : "#38bdf8";
  const mlpCol = print ? "#16853e" : "#22c55e";
  const imuColors = print
    ? ["#2563eb", "#16a34a", "#ea580c", "#dc2626", "#9333ea", "#78716c"]
    : ["#60a5fa", "#4ade80", "#fb923c", "#f87171", "#c084fc", "#a1887f"];
  const lw = print ? 3 : 1.5;
  const fs = print ? 2 : 1; // font scale

  const pad = { l: 44 * fs, r: 16 * fs, t: 28 * fs, mid: 18 * fs, b: 22 * fs };
  const topH = 120 * fs, botH = 150 * fs;
  const plotW = W - pad.l - pad.r;

  const xMs = shot.xMs;
  const xMin = xMs[0], xMax = xMs[xMs.length - 1];
  const xScale = (x) => pad.l + (x - xMin) / (xMax - xMin) * plotW;

  ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);

  const predL = xScale(-600), predR = xScale(-100);
  const trigX = xScale(0);

  ctx.fillStyle = text; ctx.font = `bold ${10 * fs}px system-ui, sans-serif`; ctx.textAlign = "left";
  ctx.fillText("p(shoot)", pad.l, pad.t - 10 * fs);

  const legY = pad.t - 16 * fs;
  ctx.fillStyle = lrCol; ctx.fillRect(W - pad.r - 68 * fs, legY, 10 * fs, 3 * fs);
  ctx.font = `${10 * fs}px system-ui, sans-serif`;
  ctx.fillStyle = text; ctx.fillText("LR", W - pad.r - 54 * fs, legY + 4 * fs);
  ctx.fillStyle = mlpCol; ctx.fillRect(W - pad.r - 30 * fs, legY, 10 * fs, 3 * fs);
  ctx.fillStyle = text; ctx.fillText("MLP", W - pad.r - 16 * fs, legY + 4 * fs);

  const topY0 = pad.t, topY1 = pad.t + topH;
  ctx.fillStyle = band; ctx.fillRect(predL, topY0, predR - predL, topH);
  ctx.strokeStyle = trig; ctx.lineWidth = 1 * fs;
  ctx.beginPath(); ctx.moveTo(trigX, topY0); ctx.lineTo(trigX, topY1); ctx.stroke();

  ctx.fillStyle = text; ctx.font = `${9 * fs}px system-ui, sans-serif`; ctx.textAlign = "right";
  ctx.fillText("1.0", pad.l - 5 * fs, topY0 + 8 * fs);
  ctx.fillText("0.5", pad.l - 5 * fs, (topY0 + topY1) / 2 + 3 * fs);
  ctx.fillText("0.0", pad.l - 5 * fs, topY1 + 1 * fs);

  const probYScale = (p) => topY1 - p * topH;
  ctx.strokeStyle = grid; ctx.lineWidth = 0.5 * fs;
  for (const p of [0.25, 0.5, 0.75]) {
    const y = probYScale(p);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
  }

  function drawLine(data, color) {
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = xScale(xMs[i]), y = probYScale(data[i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  drawLine(shot.probsLR, lrCol);
  drawLine(shot.probsMLP, mlpCol);

  const botY0 = topY1 + pad.mid, botY1 = botY0 + botH;
  ctx.fillStyle = text; ctx.font = `bold ${10 * fs}px system-ui, sans-serif`; ctx.textAlign = "left";
  ctx.fillText("IMU (raw i16)", pad.l, botY0 - 6 * fs);

  const imuLabels = ["ax", "ay", "az", "gx", "gy", "gz"];
  ctx.font = `${9 * fs}px system-ui, sans-serif`;
  let legX = W - pad.r;
  for (let ch = 5; ch >= 0; ch--) {
    const tw = ctx.measureText(imuLabels[ch]).width;
    legX -= tw;
    ctx.fillStyle = imuColors[ch]; ctx.fillText(imuLabels[ch], legX, botY0 - 6 * fs);
    legX -= 12 * fs;
  }

  ctx.fillStyle = band; ctx.fillRect(predL, botY0, predR - predL, botH);
  ctx.strokeStyle = trig; ctx.lineWidth = 1 * fs;
  ctx.beginPath(); ctx.moveTo(trigX, botY0); ctx.lineTo(trigX, botY1); ctx.stroke();

  let imuMin = Infinity, imuMax = -Infinity;
  for (const arr of [shot.axW, shot.ayW, shot.azW, shot.gxW, shot.gyW, shot.gzW])
    for (const v of arr) { if (v < imuMin) imuMin = v; if (v > imuMax) imuMax = v; }
  const imuPad2 = (imuMax - imuMin) * 0.05 + 1;
  imuMin -= imuPad2; imuMax += imuPad2;
  const imuYScale = (v) => botY1 - (v - imuMin) / (imuMax - imuMin) * botH;

  const imuData = [shot.axW, shot.ayW, shot.azW, shot.gxW, shot.gyW, shot.gzW];
  for (let ch = 0; ch < 6; ch++) {
    ctx.strokeStyle = imuColors[ch]; ctx.lineWidth = (print ? 1.5 : 0.8) * fs; ctx.globalAlpha = print ? 0.85 : 0.7; ctx.beginPath();
    for (let i = 0; i < imuData[ch].length; i++) {
      const x = xScale(xMs[i]), y = imuYScale(imuData[ch][i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  ctx.fillStyle = text; ctx.font = `${9 * fs}px system-ui, sans-serif`; ctx.textAlign = "center";
  const step = 200;
  for (let ms = Math.ceil(xMin / step) * step; ms <= xMax; ms += step) {
    const x = xScale(ms);
    ctx.fillText(ms + "ms", x, botY1 + 14 * fs);
  }
}

// ── Results display ─────────────────────────────────────────────────────────

function fmtPct(x) { return `${(x * 100).toFixed(1)}%`; }

function drawConfusionMatrix(canvas, cm) {
  const dpr = window.devicePixelRatio || 1;
  const W = 520, H = 240;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.classList.add("cmCanvas");
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  ctx.fillStyle = "#0b0e14";
  ctx.fillRect(0, 0, W, H);

  const pad = 18;
  const gridX0 = pad + 70;
  const gridY0 = pad + 30;
  const cellW = 170;
  const cellH = 80;

  const tn = cm.tn ?? 0, fp = cm.fp ?? 0, fn = cm.fn ?? 0, tp = cm.tp ?? 0;
  const vals = [tn, fp, fn, tp];
  const vmax = Math.max(1, ...vals);

  function cellColor(v) {
    const a = Math.min(1, v / vmax);
    return `rgba(56,189,248,${0.12 + 0.55 * a})`;
  }

  // Labels
  ctx.fillStyle = "#9aa4b2";
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("Confusion matrix (test)", pad, pad + 4);

  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText("Pred 0", gridX0 + 10, gridY0 - 10);
  ctx.fillText("Pred 1", gridX0 + cellW + 10, gridY0 - 10);
  ctx.save();
  ctx.translate(gridX0 - 40, gridY0 + cellH);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("True 0", 0, 0);
  ctx.restore();
  ctx.save();
  ctx.translate(gridX0 - 40, gridY0 + cellH + cellH);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("True 1", 0, 0);
  ctx.restore();

  // Cells: [ [tn, fp], [fn, tp] ]
  const cells = [
    { x: gridX0, y: gridY0, v: tn, label: "TN" },
    { x: gridX0 + cellW, y: gridY0, v: fp, label: "FP" },
    { x: gridX0, y: gridY0 + cellH, v: fn, label: "FN" },
    { x: gridX0 + cellW, y: gridY0 + cellH, v: tp, label: "TP" },
  ];
  ctx.textAlign = "left";
  for (const c of cells) {
    ctx.fillStyle = cellColor(c.v);
    ctx.fillRect(c.x, c.y, cellW - 10, cellH - 10);
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.strokeRect(c.x, c.y, cellW - 10, cellH - 10);
    ctx.fillStyle = "rgba(229,231,235,0.95)";
    ctx.font = "bold 18px system-ui, sans-serif";
    ctx.fillText(String(c.v), c.x + 12, c.y + 34);
    ctx.fillStyle = "#9aa4b2";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText(c.label, c.x + 12, c.y + 56);
  }
}

function renderMetrics(metrics) {
  metricsEl.replaceChildren();
  if (!metrics?.lr || !metrics?.mlp) {
    const div = document.createElement("div");
    div.className = "muted";
    div.textContent = "Not enough data to compute a reliable held-out metric check yet. Record a longer session with more shots and more non-shooting aiming (negatives).";
    metricsEl.appendChild(div);
    return;
  }

  const mkCard = (title, subtitle, m) => {
    const card = document.createElement("div");
    card.className = "metricCard";

    const head = document.createElement("div");
    head.className = "metricHead";
    const t = document.createElement("div");
    t.className = "metricTitle";
    t.textContent = title;
    const sub = document.createElement("div");
    sub.className = "metricSub";
    sub.textContent = subtitle;
    head.appendChild(t);
    head.appendChild(sub);
    card.appendChild(head);

    const row = document.createElement("div");
    row.className = "metricRow";
    const pills = [
      `Precision ${fmtPct(m.precision)}`,
      `Recall ${fmtPct(m.recall)}`,
      `F1 ${fmtPct(m.f1)}`,
      `Acc ${fmtPct(m.acc)}`,
      `Thr ${(m.threshold ?? 0.5).toFixed(2)} (best ${(m.bestThreshold ?? 0.5).toFixed(2)})`,
    ];
    for (const txt of pills) {
      const p = document.createElement("span");
      p.className = "metricPill";
      p.textContent = txt;
      row.appendChild(p);
    }
    card.appendChild(row);

    const canvas = document.createElement("canvas");
    drawConfusionMatrix(canvas, m.cm ?? { tp: 0, tn: 0, fp: 0, fn: 0 });
    card.appendChild(canvas);

    return card;
  };

  metricsEl.appendChild(mkCard("LR", "18 features · fast & stable", metrics.lr));
  metricsEl.appendChild(mkCard("MLP", "30 features · more expressive", metrics.mlp));

  const help = document.createElement("div");
  help.className = "callout";
  const b = document.createElement("b");
  b.textContent = "How to tell it's good:";
  help.appendChild(b);
  help.appendChild(document.createElement("br"));
  help.appendChild(document.createTextNode("You want high precision (few false spin-ups) and high recall (early spin-ups before real shots). If either model looks bad, record longer with more non-shooting aiming/movement (negatives)."));
  metricsEl.appendChild(help);
}

function renderDataQuality(result) {
  dataQualityEl.replaceChildren();
  const s = result.summary;
  const recSec = s.durationMs / 1000;
  const rawShots = s.rawPosCount ?? s.shotsAccepted;

  // Score components (0-100 each)
  const shotScore = Math.min(100, Math.round((rawShots / 100) * 100)); // 100 shots = 100%
  const timeScore = Math.min(100, Math.round((recSec / 300) * 100)); // 5 min = 100%
  const negRatio = s.negWindows / Math.max(1, s.posWindows);
  const negScore = Math.min(100, Math.round(Math.min(negRatio, 1) * 100));
  const overall = Math.round((shotScore + timeScore + negScore) / 3);

  const tier = overall >= 70 ? "good" : overall >= 40 ? "fair" : "bad";
  const label = overall >= 70 ? "Good" : overall >= 40 ? "Fair" : "Needs more data";

  const wrap = document.createElement("div");
  wrap.className = `dq-wrap dq-${tier}`;

  const scoreDiv = document.createElement("div");
  scoreDiv.className = "dq-score";
  scoreDiv.textContent = overall;
  wrap.appendChild(scoreDiv);

  const details = document.createElement("div");
  details.className = "dq-details";
  const b = document.createElement("div");
  b.className = "dq-label";
  b.textContent = `Data quality: ${label}`;
  details.appendChild(b);
  const bars = [
    { label: "Shots", score: shotScore, tip: `${rawShots} shots (aim for 100+)` },
    { label: "Duration", score: timeScore, tip: `${recSec.toFixed(0)}s (aim for 5+ min)` },
    { label: "Negatives", score: negScore, tip: `${negRatio.toFixed(1)}:1 neg/pos ratio (aim for 1:1+)` },
  ];
  for (const bar of bars) {
    const row = document.createElement("div");
    row.className = "dq-bar-row";
    const lbl = document.createElement("span");
    lbl.className = "dq-bar-label";
    lbl.textContent = bar.label;
    const track = document.createElement("div");
    track.className = "dq-bar-track";
    const fill = document.createElement("div");
    const bt = bar.score >= 70 ? "good" : bar.score >= 40 ? "fair" : "bad";
    fill.className = `dq-bar-fill dq-${bt}`;
    fill.style.setProperty("--bar-pct", bar.score + "%");
    track.appendChild(fill);
    const tip = document.createElement("span");
    tip.className = "dq-bar-tip";
    tip.textContent = bar.tip;
    row.appendChild(lbl); row.appendChild(track); row.appendChild(tip);
    details.appendChild(row);
  }
  wrap.appendChild(details);
  dataQualityEl.appendChild(wrap);
}

function drawLossCurve(lrLoss, mlpLoss, metrics) {
  if (!lossCanvasEl) return;
  const dpr = window.devicePixelRatio || 1;
  const W = 600, H = 220;
  lossCanvasEl.width = W * dpr; lossCanvasEl.height = H * dpr;
  const ctx = lossCanvasEl.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.fillStyle = "#0b0e14"; ctx.fillRect(0, 0, W, H);

  const pad = { l: 50, r: 16, t: 20, b: 28 };
  const pW = W - pad.l - pad.r, pH = H - pad.t - pad.b;

  // Combine all loss values to find range
  const allPts = [...(lrLoss || []), ...(mlpLoss || [])];

  // Also include train/val loss from held-out eval if available
  const lrTL = metrics?.lr?.trainLoss, lrVL = metrics?.lr?.valLoss;
  const mlpTL = metrics?.mlp?.trainLoss, mlpVL = metrics?.mlp?.valLoss;
  const extraVals = [lrTL, lrVL, mlpTL, mlpVL].filter(v => v != null && Number.isFinite(v));

  if (allPts.length < 2 && extraVals.length === 0) {
    ctx.fillStyle = "#9aa4b2"; ctx.font = "13px system-ui"; ctx.fillText("Not enough data for loss curve", pad.l, H / 2);
    return;
  }

  let maxIter = 0, maxLoss = 0, minLoss = Infinity;
  for (const p of allPts) {
    if (p.iter > maxIter) maxIter = p.iter;
    if (p.loss > maxLoss) maxLoss = p.loss;
    if (p.loss < minLoss) minLoss = p.loss;
  }
  for (const v of extraVals) {
    if (v > maxLoss) maxLoss = v;
    if (v < minLoss) minLoss = v;
  }
  const lPad = (maxLoss - minLoss) * 0.1 + 0.01;
  minLoss = Math.max(0, minLoss - lPad); maxLoss += lPad;

  const xScale = (iter) => pad.l + (maxIter > 0 ? (iter / maxIter) * pW : 0);
  const yScale = (loss) => pad.t + pH - ((loss - minLoss) / (maxLoss - minLoss)) * pH;

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (pH * i) / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
  }

  // Y-axis labels
  ctx.fillStyle = "#9aa4b2"; ctx.font = "9px system-ui"; ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const v = maxLoss - ((maxLoss - minLoss) * i) / 4;
    ctx.fillText(v.toFixed(2), pad.l - 5, pad.t + (pH * i) / 4 + 3);
  }

  // Draw training loss curves
  function drawCurve(pts, color) {
    if (!pts || pts.length < 2) return;
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const x = xScale(pts[i].iter), y = yScale(pts[i].loss);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  drawCurve(lrLoss, "#38bdf8");
  drawCurve(mlpLoss, "#22c55e");

  // Draw train/val loss markers from held-out eval (horizontal dashed lines at right edge)
  function drawLossMarker(loss, color, label, yOff) {
    if (loss == null || !Number.isFinite(loss)) return;
    const y = yScale(loss);
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = color; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(W - pad.r - 120, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = color; ctx.font = "9px system-ui"; ctx.textAlign = "right";
    ctx.fillText(label + " " + loss.toFixed(3), W - pad.r - 2, y + yOff);
  }
  drawLossMarker(lrTL, "#38bdf880", "LR train", -4);
  drawLossMarker(lrVL, "#38bdf8", "LR val", 10);
  drawLossMarker(mlpTL, "#22c55e80", "MLP train", -4);
  drawLossMarker(mlpVL, "#22c55e", "MLP val", 10);

  // Legend
  ctx.fillStyle = "#38bdf8"; ctx.fillRect(pad.l + 8, pad.t + 4, 12, 3);
  ctx.fillStyle = "#9aa4b2"; ctx.font = "10px system-ui"; ctx.textAlign = "left";
  ctx.fillText("LR train", pad.l + 24, pad.t + 8);
  ctx.fillStyle = "#22c55e"; ctx.fillRect(pad.l + 78, pad.t + 4, 12, 3);
  ctx.fillStyle = "#9aa4b2"; ctx.fillText("MLP train", pad.l + 94, pad.t + 8);
  ctx.fillStyle = "#9aa4b2"; ctx.fillText("dashed = held-out train/val", pad.l + 160, pad.t + 8);

  // X-axis
  ctx.fillStyle = "#9aa4b2"; ctx.font = "9px system-ui"; ctx.textAlign = "center";
  ctx.fillText("Epoch", W / 2, H - 4);
}

function showResults(result) {
  resultsEl.classList.remove("hidden");
  renderDataQuality(result);
  drawLossCurve(result.lrLossHistory, result.mlpLossHistory, result.metrics);
  const s = result.summary;
  summaryEl.replaceChildren();
  const trainSec = result.trainingMs != null ? (result.trainingMs / 1000).toFixed(1) : null;
  const rows = [
    ["Recording:", `${(s.durationMs / 1000).toFixed(1)}s (${s.samples} samples)`],
    ["Shots:", `${s.shotsAll} total, ${s.shotsAccepted} accepted`],
    ["Training windows:", `${s.posWindows} positive, ${s.negWindows} negative`],
    ...(trainSec ? [["Trained in:", `${trainSec}s (LR + MLP, in-browser)`]] : []),
  ];
  for (const [k, v] of rows) {
    const div = document.createElement("div");
    const b = document.createElement("b"); b.textContent = k;
    div.appendChild(b);
    div.appendChild(document.createTextNode(" " + v));
    summaryEl.appendChild(div);
  }

  if (result.warnings && result.warnings.length > 0) {
    const warnDiv = document.createElement("div");
    warnDiv.className = "dq-warning";
    warnDiv.innerHTML = "<b>Heads up:</b> " + result.warnings.join(" ");
    summaryEl.appendChild(warnDiv);
  }

  renderMetrics(result.metrics);

  shotGridEl.replaceChildren();
  const shots = (result.shotPlots ?? []).slice(0, 2);
  if (shots.length === 0) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.textContent = "No good shots found to display (no clear low→high prediction). Record a longer session with more shots and more non-shooting aiming.";
    shotGridEl.appendChild(empty);
  }
  for (const shot of shots) {
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
    setTrainingViz(true);
    await sleep(30); // allow overlay to paint before heavy work
    mlpViz.setSubtitle("Preparing dataset…");
    const trainStart = performance.now();
    const result = await trainPipeline(logBytes, log, {
      onMlpSnapshot: (snap) => {
        // snap contains (iter/maxIter + weights)
        mlpViz.updateSnapshot(snap);
      },
      // Make the experience visible even on fast machines.
      mlpSnapshotEvery: 8,
      mlpDelayMs: 18,
      mlpMaxIter: 700,
    });
    const minVizMs = 5200;
    const elapsed = performance.now() - trainStart;
    if (elapsed < minVizMs) await sleep(minVizMs - elapsed);
    setTrainingViz(false);
    result.trainingMs = elapsed;
    trainResult = result; status[2] = "ok";

    status[3] = "ok"; status[4] = "ok";
    renderSteps(4, status); setSpinner(false);
    log(`Training complete in ${(elapsed / 1000).toFixed(1)}s.`);

    showResults(result);
    btnLoadModel.disabled = false; btnDownloadModels.disabled = false; btnPrintPlots.disabled = false;
    btnDownloadLog.disabled = false; btnReady.disabled = false;
    resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    setSpinner(false);
    setTrainingViz(false);
    log("ERROR: " + (e?.message ?? String(e)));
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

btnPrintPlots.addEventListener("click", () => {
  if (!trainResult?.shotPlots) return;
  const shots = trainResult.shotPlots.slice(0, 2);
  for (let i = 0; i < shots.length; i++) {
    const c = document.createElement("canvas");
    drawShotPlot(c, shots[i], true);
    c.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url; a.download = `shot_${i + 1}_print.png`; a.click();
      URL.revokeObjectURL(url);
    }, "image/png");
  }
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

// ── Import .bin logs ────────────────────────────────────────────────────────

btnImportLogs.addEventListener("click", () => { fileInputEl.click(); });

fileInputEl.addEventListener("change", async (e) => {
  const files = Array.from(e.target.files);
  if (!files.length) return;
  fileInputEl.value = ""; // reset so same files can be re-selected

  logEl.textContent = "";
  resultsEl.classList.add("hidden"); afterEl.classList.add("hidden");
  btnLoadModel.disabled = true; btnDownloadModels.disabled = true; btnDownloadLog.disabled = true;
  trainResult = null; rawLogBytes = null;

  const status = {};
  try {
    setSpinner(true);
    importStatusEl.textContent = `Loading ${files.length} file(s)...`;
    log(`Importing ${files.length} .bin file(s)...`);

    // Read and concatenate all files
    const buffers = [];
    let totalBytes = 0;
    for (const f of files) {
      const ab = await f.arrayBuffer();
      const u8 = new Uint8Array(ab);
      // Validate: must be a multiple of 17 bytes (MlSample size)
      if (u8.length % 17 !== 0) {
        log(`WARNING: ${f.name} size (${u8.length}) not a multiple of 17 bytes/sample — skipping`);
        continue;
      }
      log(`Loaded ${f.name}: ${u8.length} bytes (${Math.floor(u8.length / 17)} samples)`);
      buffers.push(u8);
      totalBytes += u8.length;
    }
    if (!buffers.length) throw new Error("No valid .bin files loaded");

    // Concatenate into one buffer
    const combined = new Uint8Array(totalBytes);
    let offset = 0;
    for (const buf of buffers) {
      combined.set(buf, offset);
      offset += buf.length;
    }
    rawLogBytes = combined;
    const totalSamples = Math.floor(totalBytes / 17);
    log(`Combined: ${totalBytes} bytes, ${totalSamples} samples from ${buffers.length} file(s)`);
    importStatusEl.textContent = `${totalSamples} samples loaded`;

    status[0] = "ok"; status[1] = "ok";
    status[2] = "run"; renderSteps(2, status);

    await sleep(50);
    setTrainingViz(true);
    await sleep(30);
    mlpViz.setSubtitle("Preparing dataset…");
    const trainStart = performance.now();
    const result = await trainPipeline(combined, log, {
      onMlpSnapshot: (snap) => mlpViz.updateSnapshot(snap),
      mlpSnapshotEvery: 8,
      mlpDelayMs: 18,
      mlpMaxIter: 700,
    });
    const minVizMs = 5200;
    const elapsed = performance.now() - trainStart;
    if (elapsed < minVizMs) await sleep(minVizMs - elapsed);
    setTrainingViz(false);
    result.trainingMs = elapsed;
    trainResult = result;

    status[2] = "ok"; status[3] = "ok"; status[4] = "ok";
    renderSteps(4, status); setSpinner(false);
    log(`Training complete in ${(elapsed / 1000).toFixed(1)}s.`);

    showResults(result);
    btnDownloadModels.disabled = false;
    btnDownloadLog.disabled = false;
    btnPrintPlots.disabled = false;
    // Model upload only available if serial is connected
    btnLoadModel.disabled = !serialIO;
    resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    setSpinner(false);
    setTrainingViz(false);
    log("ERROR: " + (err?.message ?? String(err)));
    importStatusEl.textContent = "Error — see log";
  }
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
