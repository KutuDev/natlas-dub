(() => {
  const $ = (id) => document.getElementById(id);
  const KEYS = {
    yo: ["ẹ","Ẹ","ọ","Ọ","ṣ","Ṣ","á","à","é","è","í","ì","ó","ò","ú","ù","ẹ́","ẹ̀","ọ́","ọ̀","ń"],
    ha: ["ɓ","Ɓ","ɗ","Ɗ","ƙ","Ƙ","ƴ","Ƴ"],
    ig: ["ị","Ị","ọ","Ọ","ụ","Ụ","ṅ","Ṅ"],
    en: [],
  };
  let jobId = null, job = null, lastField = null, workerOk = false;

  function setMsg(el, text, kind) { el.textContent = text || ""; el.className = "msg" + (kind ? " " + kind : ""); }
  function unlock(n) { for (let i = 1; i <= n; i++) $("step" + i).classList.remove("locked"); }
  function lockFrom(n) { for (let i = n; i <= 6; i++) $("step" + i).classList.add("locked"); }

  const OFFLINE = "GPU worker offline — processing unavailable";
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  // Render free instances sleep; a request during wake-up can fail or return 502/503
  // from Render's edge. Retry idempotent calls (and any network failure before the
  // request reached us) with backoff, and tell the user what is happening.
  async function api(path, opts = {}, tries = 0) {
    const method = (opts.method || "GET").toUpperCase();
    const retryable = method === "GET" || method === "PUT";
    let res;
    try { res = await fetch(path, opts); }
    catch (e) {
      if (tries < 6 && !(opts.body instanceof FormData)) { wakeNotice(true); await sleep(5000); return api(path, opts, tries + 1); }
      wakeNotice(false); throw new Error("Cannot reach the server. It may be waking up; please try again in a minute.");
    }
    let body = null;
    try { body = await res.json(); } catch (_) {}
    const isWorkerOffline = body && body.code === "worker_offline";
    if ((res.status === 502 || res.status === 503) && !isWorkerOffline && !(body && body.code) && retryable && tries < 6) {
      wakeNotice(true); await sleep(5000); return api(path, opts, tries + 1);
    }
    wakeNotice(false);
    if (isWorkerOffline) showOffline(body.detail);
    if (!res.ok) throw new Error((body && body.detail) ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : ("HTTP " + res.status));
    return body;
  }
  function wakeNotice(on) {
    const b = $("workerBadge");
    if (on) { b.textContent = "Waking up server…"; b.className = "badge"; }
  }
  function showOffline(detail) {
    workerOk = false;
    const b = $("workerBadge"); b.textContent = "GPU worker offline"; b.className = "badge bad";
    const ban = $("offlineBanner");
    ban.textContent = OFFLINE + ". Transcription, translation, voice and rendering all run on the GPU worker, which is not connected right now. Nothing will be faked; please check back shortly." + (detail && detail !== OFFLINE ? "" : "");
    ban.classList.remove("hidden");
  }
  const jsonPost = (path, data, method = "POST") => api(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(data || {}) });

  // ---------------- health
  async function checkHealth() {
    try {
      const h = await api("/health");
      $("limitS").textContent = h.limits.max_video_seconds; $("limitMB").textContent = h.limits.max_video_mb;
      const w = h.gpu_worker || {};
      workerOk = !!w.online;
      const b = $("workerBadge");
      if (workerOk) {
        b.textContent = w.ready ? "GPU worker online" : "GPU worker online (models loading)";
        b.className = w.ready ? "badge ok" : "badge";
        $("offlineBanner").classList.add("hidden");
        if (w.limits) { $("limitS").textContent = w.limits.max_video_seconds; $("limitMB").textContent = w.limits.max_video_mb; }
      } else showOffline();
    } catch (e) { $("workerBadge").textContent = "Server unreachable"; $("workerBadge").className = "badge bad"; }
    $("uploadBtn").disabled = !workerOk;
  }
  checkHealth(); setInterval(checkHealth, 15000);

  // ---------------- polling
  async function poll(msgEl) {
    for (;;) {
      job = await api(`/api/jobs/${jobId}`);
      if (job.status === "running") { setMsg(msgEl, job.message || "Working…", "busy"); await sleep(2000); continue; }
      if (job.status === "error") {
        throw new Error(job.error);
      }
      setMsg(msgEl, job.message || "Done", "ok");
      return job;
    }
  }

  function busy(btn, on) { btn.disabled = on; }

  // ---------------- step 1
  $("uploadBtn").onclick = async () => {
    const f = $("file").files[0];
    if (!f) return setMsg($("uploadMsg"), "Choose a video first.", "err");
    if (!$("consent").checked) return setMsg($("uploadMsg"), "Please tick the rights/consent box.", "err");
    const fd = new FormData();
    fd.append("file", f); fd.append("consent", "true");
    fd.append("tester_name", $("testerName").value); fd.append("tester_email", $("testerEmail").value);
    busy($("uploadBtn"), true); setMsg($("uploadMsg"), "Uploading…", "busy"); lockFrom(2);
    try {
      const r = await api("/api/upload", { method: "POST", body: fd });
      jobId = r.job_id;
      setMsg($("uploadMsg"), `Uploaded: ${r.duration}s, ${r.size_mb} MB.`, "ok");
      const v = $("sourceVideo"); v.src = `/api/jobs/${jobId}/source`; v.classList.remove("hidden");
      $("segments").innerHTML = ""; $("reviewActions").classList.add("hidden");
      $("audioPreview").classList.add("hidden"); $("resultVideo").classList.add("hidden");
      unlock(3);
    } catch (e) { setMsg($("uploadMsg"), e.message, "err"); }
    busy($("uploadBtn"), false);
  };

  // ---------------- step 3
  function buildKeyboard() {
    const lang = $("targetLang").value, kb = $("keyboard");
    kb.innerHTML = "";
    if (!KEYS[lang].length) return kb.classList.add("hidden");
    kb.classList.remove("hidden");
    const label = document.createElement("span"); label.className = "hint"; label.textContent = "Tap to insert at cursor: "; kb.appendChild(label);
    KEYS[lang].forEach(ch => {
      const b = document.createElement("button"); b.type = "button"; b.textContent = ch;
      b.onmousedown = (e) => e.preventDefault();
      b.onclick = () => insertAtCursor(ch);
      kb.appendChild(b);
    });
  }
  function insertAtCursor(ch) {
    const t = lastField; if (!t) return;
    const s = t.selectionStart, e = t.selectionEnd;
    t.value = t.value.slice(0, s) + ch + t.value.slice(e);
    t.selectionStart = t.selectionEnd = s + ch.length; t.focus();
    t.dispatchEvent(new Event("input"));
  }

  const fmt = (s) => { const m = Math.floor(s / 60); return `${m}:${(s % 60).toFixed(1).padStart(4, "0")}`; };

  function renderSegments() {
    const box = $("segments"); box.innerHTML = "";
    job.segments.forEach(seg => {
      const d = document.createElement("div"); d.className = "seg" + (seg.status === "approved" ? " approved" : "");
      const flags = (seg.flags || []).map(f => `<div class="flag">⚠ Low-confidence transcription: ${f.replace(/_/g, " ")}</div>`).join("");
      const warns = (seg.warnings || []).map(w => `<div class="flag">⚠ ${esc(w)}</div>`).join("");
      d.innerHTML = `
        <div class="seghead"><span>#${seg.index + 1} · ${fmt(seg.start)} → ${fmt(seg.end)}</span><span class="status ${seg.status}">${seg.status.replace("_", " ")}</span></div>
        <div class="orig"><b>Original:</b> ${esc(seg.original_text)}</div>
        <div class="ai"><b>AI translation:</b> ${esc(seg.translated_text)}</div>${flags}${warns}
        <textarea lang="${$("targetLang").value}"></textarea>
        <div class="segfoot"><span class="cc">${seg.char_count} chars</span><button class="approve">${seg.status === "approved" ? "Approved ✓" : "Approve"}</button></div>`;
      const ta = d.querySelector("textarea"); ta.value = seg.final_text;
      ta.onfocus = () => { lastField = ta; };
      ta.oninput = () => { d.querySelector(".cc").textContent = ta.value.length + " chars"; };
      ta.onchange = () => saveSeg(seg, ta, d);
      d.querySelector(".approve").onclick = async () => {
        try {
          if (ta.value.trim() !== seg.final_text) await saveSeg(seg, ta, d);
          const r = await jsonPost(`/api/jobs/${jobId}/segments/${seg.index}/approve`);
          Object.assign(seg, r); refreshSeg(d, seg); updateGate();
        } catch (e) { alert(e.message); }
      };
      box.appendChild(d);
    });
    $("reviewActions").classList.remove("hidden");
    $("srtLink").href = `/api/jobs/${jobId}/srt`; $("downloadSrt").href = `/api/jobs/${jobId}/srt`;
    updateGate();
  }
  function refreshSeg(d, seg) {
    d.className = "seg" + (seg.status === "approved" ? " approved" : "");
    const st = d.querySelector(".status"); st.className = "status " + seg.status; st.textContent = seg.status.replace("_", " ");
    d.querySelector(".approve").textContent = seg.status === "approved" ? "Approved ✓" : "Approve";
  }
  async function saveSeg(seg, ta, d) {
    const r = await jsonPost(`/api/jobs/${jobId}/segments/${seg.index}`, { edited_text: ta.value }, "PUT");
    Object.assign(seg, r); refreshSeg(d, seg); lockFrom(5);
    $("audioPreview").classList.add("hidden"); updateGate();
  }
  function updateGate() {
    const allOk = job.segments.every(s => s.status === "approved" || !s.final_text);
    if (allOk) unlock(4); else lockFrom(4);
  }
  const esc = (s) => String(s || "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  $("targetLang").onchange = buildKeyboard;
  $("processBtn").onclick = async () => {
    const terms = $("protectedTerms").value.split(",").map(s => s.trim()).filter(Boolean);
    busy($("processBtn"), true); lockFrom(4); $("segments").innerHTML = "";
    try {
      await jsonPost(`/api/jobs/${jobId}/process`, { source_language: $("sourceLang").value, target_language: $("targetLang").value, voice: $("voice").value, protected_terms: terms });
      await poll($("processMsg"));
      buildKeyboard(); renderSegments();
    } catch (e) { setMsg($("processMsg"), e.message, "err"); }
    busy($("processBtn"), false);
  };
  $("approveAllBtn").onclick = async () => {
    // Save any unsaved edits first.
    for (const d of $("segments").querySelectorAll(".seg")) {
      const ta = d.querySelector("textarea"); const idx = [...$("segments").children].indexOf(d); const seg = job.segments[idx];
      if (ta.value.trim() !== seg.final_text) await saveSeg(seg, ta, d);
    }
    job = await jsonPost(`/api/jobs/${jobId}/approve-all`); renderSegments();
  };

  // ---------------- step 4
  $("speed").oninput = () => { $("speedVal").textContent = Number($("speed").value).toFixed(2) + "x"; };
  $("ttsBtn").onclick = async () => {
    busy($("ttsBtn"), true); lockFrom(5); $("ttsWarnings").innerHTML = "";
    try {
      await jsonPost(`/api/jobs/${jobId}/tts`, { speed: Number($("speed").value), voice: $("voice").value });
      await poll($("ttsMsg"));
      (job.warnings || []).forEach(w => { const li = document.createElement("li"); li.textContent = w; $("ttsWarnings").appendChild(li); });
      const a = $("audioPreview"); a.src = `/api/jobs/${jobId}/audio?t=${Date.now()}`; a.classList.remove("hidden");
      unlock(5);
    } catch (e) { setMsg($("ttsMsg"), e.message, "err"); }
    busy($("ttsBtn"), false);
  };

  // ---------------- step 5/6
  $("renderBtn").onclick = async () => {
    busy($("renderBtn"), true); $("step6").classList.add("locked");
    try {
      await jsonPost(`/api/jobs/${jobId}/render`, { subtitles: $("optSubs").checked, background: $("optBg").checked });
      await poll($("renderMsg"));
      const v = $("resultVideo"); v.src = `/api/jobs/${jobId}/video?t=${Date.now()}`; v.classList.remove("hidden");
      $("downloadBtn").href = `/api/jobs/${jobId}/video?download=true`;
      unlock(6);
    } catch (e) { setMsg($("renderMsg"), e.message, "err"); }
    busy($("renderBtn"), false);
  };
  // Changing options after a render requires re-rendering before export.
  ["optSubs", "optBg"].forEach(id => $(id).onchange = () => $("step6").classList.add("locked"));
})();
