const stepsEl = document.getElementById("steps");
const logEl = document.getElementById("log");
const spinnerEl = document.getElementById("spinner");
const connEl = document.getElementById("conn");
const btnReady = document.getElementById("btnReady");
const btnDisconnect = document.getElementById("btnDisconnect");
const btnDownloadTrace = document.getElementById("btnDownloadTrace");
const resultsEl = document.getElementById("results");
const summaryEl = document.getElementById("summary");
const shotGridEl = document.getElementById("shotGrid");
const btnLoadModel = document.getElementById("btnLoadModel");
const btnDownloadModels = document.getElementById("btnDownloadModels");
const afterEl = document.getElementById("after");

const STEPS = [
  "Connect to blaster",
  "Pull log from device (MLDUMP)",
  "Upload log to trainer",
  "Train LR + MLP",
  "Generate plots",
  "Ready to load models",
];

let port = null;
let writer = null;
let serialIO = null;

let trainedModels = null; // { lr_b64, mlp_b64, summary, shots:[{png_b64}] }

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Debug trace (downloadable) to make diagnosing WebSerial issues easier.
const TRACE = []; // {t_ms, kind, msg}
const TRACE_T0 = performance.now();
function trace(kind, msg) {
  TRACE.push({ t_ms: Math.round(performance.now() - TRACE_T0), kind, msg });
  btnDownloadTrace.disabled = TRACE.length === 0;
}

function apiBase() {
  // If opened as file://, location.origin is "null" and relative fetches fail.
  if (location.origin === "null") return "http://127.0.0.1:8000";
  return "";
}

function apiUrl(path) {
  return apiBase() + path;
}

function setSpinner(on) {
  spinnerEl.classList.toggle("hidden", !on);
}

function log(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
  trace("log", msg);
}

function renderSteps(activeIdx, statusByIdx = {}) {
  stepsEl.innerHTML = "";
  STEPS.forEach((s, i) => {
    const row = document.createElement("div");
    row.className = "step";
    const left = document.createElement("div");
    left.textContent = s;
    const tag = document.createElement("div");
    tag.className = "tag";
    const st = statusByIdx[i] ?? (i < activeIdx ? "ok" : i === activeIdx ? "run" : "todo");
    if (st === "ok") {
      tag.textContent = "done";
      tag.classList.add("ok");
    } else if (st === "run") {
      tag.textContent = "running";
      tag.classList.add("run");
    } else if (st === "err") {
      tag.textContent = "error";
      tag.classList.add("err");
    } else {
      tag.textContent = "pending";
    }
    row.appendChild(left);
    row.appendChild(tag);
    stepsEl.appendChild(row);
  });
}

async function ensureSerial() {
  if (!("serial" in navigator)) {
    throw new Error("Web Serial is not supported in this browser. Use Chrome/Edge.");
  }
}

async function connect() {
  await ensureSerial();
  port = await navigator.serial.requestPort();
  await port.open({ baudRate: 115200 });
  // Some USB-CDC stacks care about DTR/RTS for command processing.
  try {
    await port.setSignals({ dataTerminalReady: true, requestToSend: true });
  } catch {}
  writer = port.writable.getWriter();
  serialIO = new SerialIO(port);
  // Drop any pre-existing boot/menu chatter so the first command starts cleanly.
  serialIO.clear();
  trace("serial", "port opened + DTR/RTS asserted");
  btnDisconnect.disabled = false;
  connEl.textContent = "Connected";
  log("Connected to serial.");
}

async function disconnect() {
  try {
    // Drop DTR/RTS so firmware can treat this as "disconnected".
    try {
      if (port) await port.setSignals({ dataTerminalReady: false, requestToSend: false });
    } catch {}
    trace("serial", "DTR/RTS deasserted");

    if (serialIO) await serialIO.close();
    if (writer) {
      try {
        await writer.close();
      } catch {}
      writer.releaseLock();
    }
    if (port) {
      await port.close();
    }
  } finally {
    port = null;
    writer = null;
    serialIO = null;
    btnDisconnect.disabled = true;
    connEl.textContent = "Not connected";
    trace("serial", "port closed");
  }
}

function bytesToHex(u8) {
  return Array.from(u8)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function crc32(u8) {
  // Standard CRC32 (IEEE) like zlib.crc32
  let crc = 0xffffffff;
  for (let i = 0; i < u8.length; i++) {
    crc ^= u8[i];
    for (let j = 0; j < 8; j++) {
      const mask = -(crc & 1);
      crc = (crc >>> 1) ^ (0xedb88320 & mask);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

async function readLine(timeoutMs = 5000) {
  return serialIO.readLine(timeoutMs);
}

async function readExact(n, timeoutMs = 15000) {
  return serialIO.readExact(n, timeoutMs);
}

async function writeText(s) {
  const enc = new TextEncoder();
  await writer.write(enc.encode(s));
  trace("tx", s.replace(/\r/g, "\\r").replace(/\n/g, "\\n"));
}

// Soft resync: avoid toggling DTR/RTS (can confuse some USB-CDC stacks and firmware state machines).
async function resyncSerial(label = "Resyncing serial session...") {
  log(label);
  serialIO.clear();
  // Nudge the line-based parser toward a boundary.
  try {
    await writeText("\n");
  } catch {}
  await sleep(60);
  trace("serial", "resync: cleared buffers");
}

// Hard resync: only use as a last resort.
async function hardResyncSerial(label = "Resyncing serial session...") {
  log(label);
  if (port) {
    try {
      await port.setSignals({ dataTerminalReady: false, requestToSend: false });
    } catch {}
    await sleep(120);
    try {
      await port.setSignals({ dataTerminalReady: true, requestToSend: true });
    } catch {}
    await sleep(120);
  }
  serialIO.clear();
  trace("serial", "resync: toggled DTR/RTS + cleared buffers");
}

async function pullLog() {
  async function attemptOnce() {
    await writeText("MLDUMP\n");
    log("Sent MLDUMP");

    let size = null;
    for (let i = 0; i < 40; i++) {
      const line = await readLine(4000);
      log("<< " + line);
      if (line.startsWith("MLDUMP1 ")) {
        const n = parseInt(line.slice("MLDUMP1 ".length).trim(), 10);
        if (Number.isFinite(n) && n > 0) size = n;
        break;
      }
    }
    if (!size || size <= 0) throw new Error("Did not receive MLDUMP1 <size> header");
    log(`Expecting ${size} bytes...`);

    // Live progress: bytes received + KB/s.
    const progressT0 = performance.now();
    let lastUi = 0;
    let lastBytes = 0;
    const blob = await serialIO.readExactWithProgress(
      size,
      180000,
      (got) => {
        const now = performance.now();
        if (now - lastUi < 250) return;
        lastUi = now;
        const dt = (now - progressT0) / 1000;
        const kbps = dt > 0 ? (got / 1024) / dt : 0;
        const inst = (got - lastBytes) / 1024 / Math.max(0.001, (now - (lastUi - 250)) / 1000);
        lastBytes = got;
        log(`MLDUMP progress: ${got}/${size} bytes (${kbps.toFixed(1)} KB/s)`);
        trace("mldump", `${got}/${size} bytes`);
      },
    );
    log(`Read ${blob.length} bytes`);

    // Best-effort: wait for the trailing marker. Never fail the dump if we already got the bytes.
    const doneDeadline = performance.now() + 1500;
    while (performance.now() < doneDeadline) {
      try {
        const line = await readLine(250);
        if (line.includes("MLDUMP_DONE")) {
          log("<< MLDUMP_DONE");
          break;
        }
      } catch (e) {
        const msg = e?.message ?? String(e);
        if (!msg.includes("Timeout")) throw e;
        // keep waiting
      }
    }
    return blob;
  }

  await resyncSerial("Resyncing serial session...");
  try {
    return await attemptOnce();
  } catch (e) {
    log("MLDUMP failed, retrying once...");
    await hardResyncSerial("Resyncing and retrying MLDUMP...");
    return await attemptOnce();
  }
}

async function trainOnServer(logBytes) {
  // Preflight: verify server is reachable.
  try {
    const ping = await fetch(apiUrl("/api/ping"), { cache: "no-store" });
    if (!ping.ok) throw new Error(`ping status ${ping.status}`);
  } catch (e) {
    const origin = location.origin;
    const hint =
      origin === "null"
        ? "Open http://127.0.0.1:8000/ in Chrome/Edge (not the file directly), and ensure `python ml_web/server/app.py` is running."
        : "Ensure `python ml_web/server/app.py` is running and you opened the page from that server (http://127.0.0.1:8000/).";
    throw new Error(`Cannot reach training server (${apiUrl("/api/ping")}): ${e?.message ?? String(e)}. ${hint}`);
  }

  try {
    const resp = await fetch(apiUrl("/api/train"), {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: logBytes,
    });
    if (!resp.ok) {
      const t = await resp.text();
      throw new Error(`Training server error (${resp.status}): ${t}`);
    }
    return await resp.json();
  } catch (e) {
    throw new Error(`Failed to call trainer (${apiUrl("/api/train")}): ${e?.message ?? String(e)}`);
  }
}

function b64ToU8(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function downloadBlob(name, u8) {
  const blob = new Blob([u8], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadText(name, text) {
  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

async function drainSerial(ms) {
  const deadline = performance.now() + ms;
  while (performance.now() < deadline) {
    try {
      const line = await readLine(120);
      log("<< " + line);
    } catch (e) {
      const msg = e?.message ?? String(e);
      if (!msg.includes("Timeout")) throw e;
      await sleep(20);
    }
  }
}

async function uploadModel(cmdPrefix, u8) {
  // If the previous command printed multiple lines, drain them fully so we don't
  // misinterpret trailing output as part of the upload handshake.
  await drainSerial(300);
  const crc = crc32(u8);
  await writeText(`${cmdPrefix} ${u8.length} ${crc.toString(16).padStart(8, "0")}\n`);
  log(`Sent ${cmdPrefix} (${u8.length} bytes, crc=${crc.toString(16)})`);
  // wait for READY/ERR
  let ready = false;
  const readyDeadline = performance.now() + 15000;
  while (performance.now() < readyDeadline) {
    try {
      const line = await readLine(800);
      log("<< " + line);
      if (line.includes("MLMODEL_READY")) {
        ready = true;
        break;
      }
      if (line.includes("MLMODEL_ERR")) throw new Error(line);
    } catch (e) {
      const msg = e?.message ?? String(e);
      if (msg.includes("Timeout")) continue;
      throw e;
    }
  }
  if (!ready) {
    throw new Error(
      "No MLMODEL_READY received. If this worked before, disconnect/reconnect and try again (CDC can get desynced), and ensure the blaster is not firing while uploading.",
    );
  }
  // stream raw bytes
  const chunk = 512;
  for (let i = 0; i < u8.length; i += chunk) {
    await writer.write(u8.subarray(i, i + chunk));
  }
  log("Uploaded bytes, waiting for OK...");
  const okDeadline = performance.now() + 15000;
  while (performance.now() < okDeadline) {
    try {
      const line = await readLine(800);
      log("<< " + line);
      if (line.includes("MLMODEL_OK")) return;
      if (line.includes("MLMODEL_ERR")) throw new Error(line);
    } catch (e) {
      const msg = e?.message ?? String(e);
      if (msg.includes("Timeout")) continue;
      throw e;
    }
  }
  throw new Error("Timeout waiting for MLMODEL_OK");
}

async function loadModel(cmd) {
  await drainSerial(250);
  await writeText(cmd + "\n");
  log("Sent " + cmd);
  const deadline = performance.now() + 8000;
  while (performance.now() < deadline) {
    try {
      const line = await readLine(800);
      log("<< " + line);
      if (line.includes("MLMODEL_LOADED")) return;
      if (line.includes("MLMODEL_ERR")) throw new Error(line);
    } catch (e) {
      const msg = e?.message ?? String(e);
      if (msg.includes("Timeout")) continue;
      throw e;
    }
  }
  throw new Error("Timeout waiting for MLMODEL_LOADED");
}

async function queryModelInfo() {
  // Drain any late lines from prior commands to avoid mis-parsing.
  await drainSerial(250);
  await writeText("MLMODEL_INFO\n");
  log("Sent MLMODEL_INFO");
  const deadline = performance.now() + 4000;
  let lastRx = performance.now();
  let gotAny = false;
  let gotMarker = false;
  while (performance.now() < deadline) {
    try {
      const line = await readLine(500);
      gotAny = true;
      lastRx = performance.now();
      log("<< " + line);
      // Ignore unrelated status lines from other commands.
      if (line.includes("MLMODEL_READY") || line.includes("MLMODEL_OK") || line.includes("MLMODEL_LOADED")) {
        continue;
      }
      if (line.includes("model_source=") || line.includes("user_model_present=") || line.includes("user_has_lr=")) {
        gotMarker = true;
      }
    } catch (e) {
      const msg = e?.message ?? String(e);
      if (!msg.includes("Timeout")) throw e;
      // If we've already seen the markers and the device has been quiet for a bit,
      // consider the response complete.
      if (gotMarker && performance.now() - lastRx > 250) break;
    }
  }
  if (!gotAny || !gotMarker) {
    throw new Error(
      "No MLMODEL_INFO response. If an upload was interrupted, wait ~10s for firmware to time out or disconnect/reconnect, then try again.",
    );
  }
}

function showResults(payload) {
  resultsEl.classList.remove("hidden");
  const s = payload.summary;
  const trig = s.trigger;
  summaryEl.innerHTML = `
    <div><b>Shots:</b> ${trig.rising_edges_all} total</div>
    <div><b>Valid shots:</b> ${trig.rising_edges_accepted}</div>
    <div><b>Rejected shots:</b> ${trig.rising_edges_rejected} (${trig.rejected_reason}, min gap ${trig.min_shot_gap_ms}ms)</div>
  `;

  shotGridEl.innerHTML = "";
  payload.shots.forEach((sh, idx) => {
    const div = document.createElement("div");
    div.className = "shot";
    const img = document.createElement("img");
    img.alt = `Shot ${idx + 1}`;
    img.src = `data:image/png;base64,${sh.png_b64}`;
    div.appendChild(img);
    shotGridEl.appendChild(div);
  });
}

btnReady.addEventListener("click", async () => {
  logEl.textContent = "";
  resultsEl.classList.add("hidden");
  afterEl.classList.add("hidden");
  btnLoadModel.disabled = true;
  btnDownloadModels.disabled = true;
  trainedModels = null;
  btnReady.disabled = true;

  const status = {};
  try {
    renderSteps(0);
    setSpinner(true);

    // 0 connect
    status[0] = "run";
    renderSteps(0, status);
    await connect();
    status[0] = "ok";

    // 1 pull log
    status[1] = "run";
    renderSteps(1, status);
    const logBytes = await pullLog();
    status[1] = "ok";

    // 2 upload + train
    status[2] = "run";
    status[3] = "run";
    status[4] = "run";
    renderSteps(2, status);
    const payload = await trainOnServer(logBytes);
    trainedModels = payload;
    status[2] = "ok";
    status[3] = "ok";
    status[4] = "ok";

    // 5 ready
    status[5] = "ok";
    renderSteps(5, status);
    setSpinner(false);
    log("Training complete.");

    showResults(payload);
    btnLoadModel.disabled = false;
    btnDownloadModels.disabled = false;
    btnReady.disabled = false;
  } catch (e) {
    setSpinner(false);
    log("ERROR: " + (e?.message ?? String(e)));
    const idx = Object.keys(status).length ? Math.max(...Object.keys(status).map((x) => parseInt(x, 10))) : 0;
    status[idx] = "err";
    renderSteps(idx, status);
    btnReady.disabled = false;
  }
});

btnDisconnect.addEventListener("click", async () => {
  await disconnect();
});

btnDownloadTrace.addEventListener("click", () => {
  const meta = {
    userAgent: navigator.userAgent,
    time: new Date().toISOString(),
  };
  const payload = { meta, trace: TRACE };
  downloadText(`stinger_ml_web_trace_${Date.now()}.json`, JSON.stringify(payload, null, 2));
});

btnDownloadModels.addEventListener("click", () => {
  if (!trainedModels) return;
  downloadBlob("ml_model_lr.bin", b64ToU8(trainedModels.lr_b64));
  downloadBlob("ml_model_mlp.bin", b64ToU8(trainedModels.mlp_b64));
});

btnLoadModel.addEventListener("click", async () => {
  if (!trainedModels) return;
  try {
    btnLoadModel.disabled = true;
    setSpinner(true);
    // Sanity check that we are talking to a firmware that supports model upload.
    await queryModelInfo();
    log("Uploading LR...");
    await uploadModel("MLMODEL_PUT_LR", b64ToU8(trainedModels.lr_b64));
    log("Uploading MLP...");
    await uploadModel("MLMODEL_PUT_MLP", b64ToU8(trainedModels.mlp_b64));
    log("Loading LR...");
    await loadModel("MLMODEL_LOAD_LR");
    log("Loading MLP...");
    await loadModel("MLMODEL_LOAD_MLP");
    log("Verifying loaded weights...");
    await queryModelInfo();
    setSpinner(false);
    afterEl.classList.remove("hidden");
    afterEl.textContent =
      "Done. Unplug and try it out. Suggested tests: try aiming without shooting (should stay low), then do your usual pre-shot motions (should rise earlier). Compare ML:LR vs ML:MLP in Motor → Idling.";
  } catch (e) {
    setSpinner(false);
    btnLoadModel.disabled = false;
    log("ERROR while uploading: " + (e?.message ?? String(e)));
  }
});

class SerialIO {
  constructor(port) {
    this.port = port;
    this.reader = port.readable.getReader();
    // Buffering strategy:
    // Keep a FIFO queue of Uint8Array chunks to avoid O(n^2) copying during large reads (MLDUMP can be 100s of KB).
    this.chunks = [];
    this.headOff = 0;
    this.total = 0;
    this._waiters = [];
    this._closed = false;
    this._pumpErr = null;
    this._pumpPromise = this._pump();
  }

  _remaining() {
    return this.total;
  }

  _take(n) {
    // Consume exactly n bytes and return them as a single Uint8Array.
    const out = new Uint8Array(n);
    let outOff = 0;
    while (outOff < n) {
      const head = this.chunks[0];
      const avail = head.length - this.headOff;
      const take = Math.min(avail, n - outOff);
      out.set(head.subarray(this.headOff, this.headOff + take), outOff);
      outOff += take;
      this.headOff += take;
      this.total -= take;
      if (this.headOff >= head.length) {
        this.chunks.shift();
        this.headOff = 0;
      }
    }
    return out;
  }

  _append(chunk) {
    if (!chunk || !chunk.length) return;
    this.chunks.push(chunk);
    this.total += chunk.length;
    this._notify();
  }

  _notify() {
    const w = this._waiters;
    this._waiters = [];
    w.forEach((fn) => fn());
  }

  async _pump() {
    try {
      while (true) {
        const { value, done } = await this.reader.read();
        if (done) break;
        if (value && value.length) this._append(value);
      }
    } catch (e) {
      this._pumpErr = e;
    } finally {
      this._closed = true;
      this._notify();
    }
  }

  async close() {
    try {
      if (this.reader) {
        try {
          await this.reader.cancel();
        } catch {}
        this.reader.releaseLock();
      }
    } finally {
      this._closed = true;
      this._notify();
    }
  }

  _closedError() {
    if (this._pumpErr) return this._pumpErr;
    return new Error("Serial reader closed");
  }

  async _waitForData(timeoutMs) {
    if (this._remaining() > 0) return;
    if (this._closed) throw this._closedError();
    await Promise.race([
      new Promise((resolve) => this._waiters.push(resolve)),
      sleep(timeoutMs).then(() => {
        throw new Error("Timeout waiting for serial data");
      }),
    ]);
    if (this._closed) throw this._closedError();
  }

  _findNewlinePos() {
    // Return the number of bytes (from start of buffered stream) up to and including '\n',
    // or -1 if not present.
    let pos = 0;
    for (let ci = 0; ci < this.chunks.length; ci++) {
      const chunk = this.chunks[ci];
      const start = ci === 0 ? this.headOff : 0;
      for (let i = start; i < chunk.length; i++) {
        if (chunk[i] === 0x0a) return pos + (i - start) + 1;
      }
      pos += chunk.length - start;
    }
    return -1;
  }

  async readLine(timeoutMs = 5000) {
    const deadline = performance.now() + timeoutMs;
    while (true) {
      // Consume all complete lines currently in the buffer.
      while (true) {
        const want = this._findNewlinePos();
        if (want < 0) break;
        const lineBytes = this._take(want); // includes \n
        let s = new TextDecoder("utf-8", { fatal: false }).decode(lineBytes);
        s = s.replace(/\r/g, "").trim();
        if (s.length) return s;
        // Empty line: keep draining.
      }

      const now = performance.now();
      const left = Math.max(0, Math.floor(deadline - now));
      if (left <= 0) throw new Error("Timeout waiting for line");
      await this._waitForData(left);
    }
  }

  async readExact(n, timeoutMs = 15000) {
    const deadline = performance.now() + timeoutMs;
    const out = new Uint8Array(n);
    let off = 0;

    while (off < n) {
      const avail = this._remaining();
      if (avail > 0) {
        const take = Math.min(avail, n - off);
        out.set(this._take(take), off);
        off += take;
        continue;
      }
      const now = performance.now();
      const left = Math.max(0, Math.floor(deadline - now));
      if (left <= 0) break;
      await this._waitForData(left);
    }

    if (off !== n) throw new Error(`Timeout reading binary (${off}/${n})`);
    return out;
  }

  async readExactWithProgress(n, timeoutMs, onProgress) {
    const deadline = performance.now() + timeoutMs;
    const out = new Uint8Array(n);
    let off = 0;

    while (off < n) {
      const avail = this._remaining();
      if (avail > 0) {
        const take = Math.min(avail, n - off);
        out.set(this._take(take), off);
        off += take;
        try {
          onProgress?.(off);
        } catch {}
        continue;
      }
      const now = performance.now();
      const left = Math.max(0, Math.floor(deadline - now));
      if (left <= 0) break;
      await this._waitForData(left);
    }

    if (off !== n) throw new Error(`Timeout reading binary (${off}/${n})`);
    return out;
  }

  clear() {
    this.chunks = [];
    this.headOff = 0;
    this.total = 0;
  }
}
