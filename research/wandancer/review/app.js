"use strict";
// Adapted from OmniDance review_web's master-audio / synchronized-video controls.
const $ = s => document.querySelector(s);
const A = $("#master");
const DIMS = [["sync", "音乐"], ["motion", "动作"], ["quality", "画质"]];
const FLAGS = [["best", "最佳"], ["flicker", "闪烁"], ["collapse", "崩坏"], ["small_motion", "动作小"], ["identity_drift", "身份漂移"], ["music_unrelated", "音乐无关"]];
let M, reviews = [], tiles = [], saveQueue = Promise.resolve();
const status = text => { $("#status").textContent = text; };
const media = info => info ? "media?id=" + encodeURIComponent(info.media_id) : "";
const element = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };
const selectedCase = () => M.cases.find(c => c.case_id === $("#sample").value);
const selectedVideo = () => tiles[Number($("#frameTarget").value)]?.v;
const vids = () => tiles.filter(t => t.v).map(t => t.v);
const clock = () => ["embed", "silent"].includes($("#audioMode").value) ? selectedVideo() : A;
const finiteDuration = v => Number.isFinite(v?.duration) ? v.duration : 0;
function fps(t) { return t.info?.probe?.fps || (t.stage === "global" ? 8 : t.stage === "local" ? 30 : 16); }
async function api(path, data) {
  const options = data === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json", "X-Review-Token": M.review_token }, body: JSON.stringify(data) };
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw Error(result.reason || result.error || "请求失败");
  return result;
}
function pauseAll() { [A, ...vids()].forEach(v => v.pause()); $("#play").textContent = "播放"; }
function seekAll(seconds) {
  const t = Math.max(0, seconds);
  [A, ...vids()].forEach(v => { const d = finiteDuration(v); if (d) v.currentTime = Math.min(t, Math.max(0, d - (tiles.find(x => x.v === v) ? 0.5 / fps(tiles.find(x => x.v === v)) : 0.001))); });
  updateTime();
}
function setMasterAudio() {
  const c = selectedCase(); if (!c) return;
  const mode = $("#audioMode").value;
  const info = mode === "swap" ? c.swap_audio : c.original_audio;
  A.pause(); A.removeAttribute("src");
  if (["original", "swap"].includes(mode) && info) A.src = media(info);
  vids().forEach(v => { v.muted = !(mode === "embed" && v === selectedVideo()); });
  seekAll(0);
}
async function playAll() {
  const m = clock(); if (!m || (!m.getAttribute("src"))) return status("当前时钟/音轨不存在，请切换列或音轨。");
  const t = m.currentTime;
  [A, ...vids()].forEach(v => { v.playbackRate = Number($("#rate").value); });
  try {
    await m.play();
    vids().forEach(v => { if (v !== m && t < finiteDuration(v)) { v.currentTime = t; v.play().catch(() => status("视频尚未加载或无法播放")); } });
    $("#play").textContent = "暂停";
  } catch (error) { status("播放失败：" + error.message); }
}
function updateTime() {
  const m = clock(); if (!m) return;
  const t = m.currentTime, d = finiteDuration(m);
  $("#time").textContent = t.toFixed(3) + "s / " + d.toFixed(3) + "s";
  if (d) $("#seek").value = Math.round(t / d * 1000);
  tiles.forEach(tile => { if (tile.frame && tile.v) tile.frame.textContent = "帧 " + Math.floor(tile.v.currentTime * fps(tile)) + " · " + tile.v.currentTime.toFixed(3) + "s"; });
}
function tick() {
  const m = clock();
  if (m && !m.paused) {
    const t = m.currentTime;
    tiles.forEach(tile => { const v = tile.v; if (!v || v === m) return; const d = finiteDuration(v);
      if (!d) return;
      if (t >= d - 0.5 / fps(tile)) { v.pause(); return; }
      if (v.readyState >= 2 && Math.abs(v.currentTime - t) > 1 / fps(tile)) v.currentTime = t;
      if (v.paused) v.play().catch(() => {});
    });
    updateTime();
  }
  requestAnimationFrame(tick);
}
function step(direction) {
  pauseAll(); const t = tiles[Number($("#frameTarget").value)]; if (!t?.v) return;
  const f = fps(t), index = Math.round(t.v.currentTime * f);
  seekAll(Math.max(0, (index + direction) / f));
}
function saveForm(form, tile) {
  const reviewer = $("#reviewer").value.trim();
  if (!/^[A-Za-z0-9_-]{1,120}$/.test(reviewer)) return status("请填写字母、数字、下划线或短横线评审别名；不要填写个人身份信息。");
  const payload = { reviewer, case_id: tile.caseId, run_id: tile.run.run_id, stage: tile.stage, note: form.querySelector("textarea").value,
    flags: [...form.querySelectorAll("input:checked")].map(i => i.value) };
  DIMS.forEach(([dim]) => { const val = form.querySelector('[name="' + dim + '"]').value; payload[dim] = val ? Number(val) : null; });
  status("保存中…");
  saveQueue = saveQueue.then(async () => { const result = await api("api/review", payload); reviews.push(result.review); status("已保存到 private JSONL（" + result.review.validation.status + "）"); }).catch(error => status("保存失败：" + error.message));
}
function rating(tile) {
  const box = element("div", undefined, "rate");
  const saved = reviews.filter(r => r.reviewer === $("#reviewer").value.trim() && r.case_id === tile.caseId && r.run_id === tile.run.run_id && r.stage === tile.stage).at(-1);
  DIMS.forEach(([dim, label]) => { const row = element("label", label + " "); const select = element("select"); select.name = dim;
    select.add(new Option("未评", "")); for (let i = 1; i <= 5; i++) select.add(new Option(i, i)); select.value = saved?.[dim] || ""; row.append(select); box.append(row); select.onchange = () => saveForm(box, tile); });
  const flags = element("div", undefined, "flags");
  FLAGS.forEach(([value, label]) => { const row = element("label"); const cb = element("input"); cb.type = "checkbox"; cb.value = value; cb.checked = saved?.flags?.includes(value) || false; row.append(cb, document.createTextNode(label)); flags.append(row); cb.onchange = () => saveForm(box, tile); });
  box.append(flags); const note = element("textarea"); note.maxLength = 2000; note.placeholder = "具体动作/拍点/异常（不填写个人信息）"; note.value = saved?.note || ""; note.onchange = () => saveForm(box, tile); box.append(note);
  const save = element("button", "保存当前评分"); save.onclick = () => saveForm(box, tile); box.append(save); return box;
}
function tileElement(tile, index) {
  const box = element("div", undefined, "tile"); box.append(element("div", tile.label, "hd"));
  if (tile.info) {
    const v = element(tile.image ? "img" : "video"); v.src = media(tile.info); box.append(v);
    if (!tile.image) { tile.v = v; v.preload = "metadata"; v.muted = true; v.playsInline = true; v.onclick = () => { $("#frameTarget").value = index; pauseAll(); setMasterAudio(); }; v.onloadedmetadata = updateTime; v.onerror = () => status("媒体加载失败：" + tile.label); tile.frame = element("div", "帧 0", "meta"); box.append(tile.frame); }
  } else box.append(element("div", "尚无输出 / 未运行", "missing"));
  const p = tile.info?.probe;
  box.append(element("div", p?.status === "measured" ? [p.width && p.width + "×" + p.height, p.fps && p.fps + " fps", p.duration_seconds.toFixed(3) + "s", p.has_audio ? "含音轨" : "无音轨"].filter(Boolean).join(" · ") : "probe: " + (p?.reason || "pending"), "meta"));
  if (tile.run) {
    box.append(element("div", tile.run.model + " @ " + tile.run.version + "\nseed=" + tile.run.seed + " · " + tile.run.condition + " · run=" + tile.run.status + " · stage=" + tile.stage + "/" + tile.item.status, "meta"));
    const validation = tile.item.validation;
    box.append(element("div", "契约 " + validation.status + ": " + (validation.reason || validation.problems.join(", ") || "验证通过"), "meta " + validation.status));
    for (const field of ["working_canvas", "crop", "trim", "expected"]) if (tile.item[field]) box.append(element("div", field + ": " + JSON.stringify(tile.item[field]), "meta"));
    if (tile.item.status === "completed" && tile.info) box.append(rating(tile));
  }
  return box;
}
function render() {
  pauseAll(); const c = selectedCase(); tiles = []; $("#grid").replaceChildren(); $("#frameTarget").replaceChildren(); if (!c) return;
  tiles.push({ label: "Reference", info: c.reference, image: true }, { label: "Ground truth", info: c.gt });
  c.runs.filter(run => $("#condition").value === "all" || run.condition === $("#condition").value).forEach(run => {
    ["global", "local"].forEach(stage => { const item = run[stage]; tiles.push({ label: run.run_id + " · " + (stage === "global" ? "Global" : "raw Local"), info: item.media, item, run, stage, caseId: c.case_id }); });
  });
  tiles.forEach((tile, i) => { $("#grid").append(tileElement(tile, i)); if (tile.v) $("#frameTarget").add(new Option(tile.label, i)); });
  const local = tiles.findIndex(t => t.stage === "local" && t.v); if (local >= 0) $("#frameTarget").value = local;
  setMasterAudio();
}
async function load() {
  const old = $("#sample").value; M = await api("api/manifest"); reviews = await api("api/reviews");
  $("#sample").replaceChildren(); M.cases.forEach(c => $("#sample").add(new Option(c.case_id, c.case_id))); if (M.cases.some(c => c.case_id === old)) $("#sample").value = old;
  const oldCondition = $("#condition").value; $("#condition").replaceChildren(new Option("全部", "all"));
  [...new Set(M.cases.flatMap(c => c.runs.map(r => r.condition)))].sort().forEach(c => $("#condition").add(new Option(c, c))); if ([...$("#condition").options].some(o => o.value === oldCondition)) $("#condition").value = oldCondition;
  render(); status(M.cases.length + " 个案例；仅外部明确清单，无目录扫描或外网上传。");
}
function go(delta) { const s = $("#sample"); if (!s.options.length) return; s.selectedIndex = (s.selectedIndex + delta + s.options.length) % s.options.length; render(); }
$("#sample").onchange = render; $("#condition").onchange = render; $("#reviewer").onchange = render;
$("#prev").onclick = () => go(-1); $("#next").onclick = () => go(1);
$("#play").onclick = () => clock()?.paused ? playAll() : pauseAll(); $("#restart").onclick = () => { pauseAll(); seekAll(0); };
$("#stepB").onclick = () => step(-1); $("#stepF").onclick = () => step(1);
$("#audioMode").onchange = () => { pauseAll(); setMasterAudio(); }; $("#frameTarget").onchange = () => { pauseAll(); setMasterAudio(); };
$("#rate").onchange = () => [A, ...vids()].forEach(v => { v.playbackRate = Number($("#rate").value); });
$("#seek").oninput = () => { const m = clock(); if (m) seekAll(Number($("#seek").value) / 1000 * finiteDuration(m)); };
$("#reload").onclick = async () => { try { await api("api/reload", {}); await load(); } catch (error) { status("刷新 blocked：" + error.message); } };
document.addEventListener("ended", e => { if (e.target !== clock()) return; if ($("#loop").checked) { seekAll(0); playAll(); } else pauseAll(); }, true);
document.addEventListener("keydown", e => { if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return; if (e.key === " ") { e.preventDefault(); $("#play").click(); } else if (e.key === ",") step(-1); else if (e.key === ".") step(1); });
requestAnimationFrame(tick); load().catch(error => status("初始化 blocked：" + error.message));
