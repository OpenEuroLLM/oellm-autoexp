"""One HTML page with the whole 100M sweep: final loss against learning rate and beta / momentum,
for every batch size and budget, from the Slurm logs.

Usage (standard library only):
  python3 make_sweep_page.py --logs $OPT_SCALE_BASE/logs --out sweep_page.html

Open the file in a browser; run again whenever more jobs have finished.

What is read:
  in-training validation  final "validation loss ... on validation set" line of every cooldown
                          (2,560 sequences; exists for phase 1 and phase 2)
  fixed-set score         the scoring jobs hyb100m_rescore_*_score{1,2,4}x[_rerun]
                          (25,600 sequences, the same for every checkpoint)
A cooldown counts only if it loaded exactly the checkpoint step it asked for (README.md,
"Phase 2 reruns"); a *_rerun replaces the first attempt.
"""
import argparse
import datetime
import glob
import json
import os
import re

HP = r"((?:beta|mom)[0-9.]+|b1-[0-9.]+_b2-[0-9.]+)"
COOL = re.compile(r"hyb100m_sweep_(adam|muon|muown_classical)_lr([0-9.e-]+)_" + HP + r"_bs(\d+)_cooldown(\d)x(_rerun)?-(\d+)\.out$")
SCORE = re.compile(r"hyb100m_rescore_(adam|muon|muown_classical)_lr([0-9.e-]+)_" + HP + r"_bs(\d+)_score(\d)x(_rerun)?-(\d+)\.out$")
TRAIN_ITERS = re.compile(r"  train_iters \.+ (\d+)")
CKPT_STEP = re.compile(r"  ckpt_step \.+ (\d+)")
LOADED = re.compile(r"successfully loaded checkpoint from \S+ .* at iteration (\d+)")
LOAD_ARG = re.compile(r"^\S*\s+load \.+ (\S+)", re.M)
FINAL = r"validation loss at iteration\s+%s on validation set \| lm loss value: ([0-9.E+-]+)"


def hp_label(s):
    return s.replace("beta", "").replace("mom", "").replace("b1-", "").replace("_b2-", "/")


def order(regex):
    """Originals before reruns, each by job id: the last valid log of a run wins."""
    return lambda p: (bool(regex.search(p)[6]), int(regex.search(p)[7]))


def collect(logs):
    rows, cool_ok, bad, unfinished = {}, {}, {}, {}
    paths = [p for p in glob.glob(os.path.join(logs, "hyb100m_sweep_*_cooldown*x*-*.out")) if COOL.search(p)]
    for p in sorted(paths, key=order(COOL)):
        m = COOL.search(p)
        key = (m[1], m[5], int(m[4]), hp_label(m[3]), float(m[2]))
        run = os.path.basename(p).rsplit("-", 1)[0]
        t = open(p, errors="replace").read()
        ti, cs, ld = TRAIN_ITERS.search(t), CKPT_STEP.search(t), LOADED.search(t)
        if not (ti and cs and ld):
            unfinished[key] = "started, nothing loaded yet"
            continue
        if cs[1] != ld[1]:
            cool_ok[run] = False
            bad[key] = f"asked for step {cs[1]}, loaded step {ld[1]}"
            continue
        cool_ok[run] = True
        v = re.findall(FINAL % ti[1], t)
        if v:
            rows[("train",) + key] = float(v[-1])
            bad.pop(key, None)
            unfinished.pop(key, None)
        else:
            unfinished[key] = "training"
    paths = [p for p in glob.glob(os.path.join(logs, "hyb100m_rescore_*_score*x*-*.out")) if SCORE.search(p)]
    for p in sorted(paths, key=order(SCORE)):
        m = SCORE.search(p)
        key = (m[1], m[5], int(m[4]), hp_label(m[3]), float(m[2]))
        t = open(p, errors="replace").read()
        v = re.findall(FINAL % r"\d+", t)
        la = LOAD_ARG.search(t)
        run = os.path.basename(os.path.dirname(la[1].rstrip("/"))) if la else None
        if v and cool_ok.get(run, True):
            rows[("fixed",) + key] = float(v[-1])
    unfinished = {k: v for k, v in unfinished.items() if ("train",) + k not in rows}
    bad = {k: v for k, v in bad.items() if ("train",) + k not in rows and k not in unfinished}
    rec = lambda k: dict(opt=k[0], budget=k[1], bs=k[2], hp=k[3], lr=k[4])
    return dict(
        rows=[dict(kind=k[0], loss=v, **rec(k[1:])) for k, v in sorted(rows.items())],
        excluded=[dict(why=v, **rec(k)) for k, v in sorted(bad.items())],
        unfinished=[dict(why=v, **rec(k)) for k, v in sorted(unfinished.items())],
        generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


PAGE = r"""<meta name="claude:viewport-fit-sides" content="cover">
<title>100M Sweep Surface</title>
<style>
/* layout: controls on top, then best settings, the lr x beta small multiples, the beta cost, the numbers */
:root {
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --wash: rgba(11,11,11,0.05);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --good: #006300;
  --sans: system-ui, -apple-system, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --wash: rgba(255,255,255,0.07);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --good: #0ca30c; color-scheme: dark } }
:root[data-theme="dark"] {
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --wash: rgba(255,255,255,0.07);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --good: #0ca30c; color-scheme: dark }
body { background: var(--page); color: var(--ink); font-family: var(--sans); font-size: 14px; line-height: 1.45;
  padding-block: 20px 48px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.wrap { max-width: 1180px; margin: 0 auto; display: flex; flex-direction: column; gap: 28px; }
h1 { font-size: 22px; margin: 0; font-weight: 650; text-wrap: balance; }
h2 { font-size: 16px; margin: 0 0 4px; font-weight: 650; text-wrap: balance; }
p { margin: 0; color: var(--ink2); max-width: 78ch; }
.sub { color: var(--ink2); margin-top: 4px; }
.controls { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: center; }
.ctl { display: flex; align-items: center; gap: 8px; }
.ctl span { color: var(--muted); font-size: 12px; letter-spacing: .04em; text-transform: uppercase; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; color: var(--ink2); background: none; border: 0; padding: 6px 12px; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--surface); }
.seg button:focus-visible { outline: 2px solid var(--s1); outline-offset: -2px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 10px 0 4px; color: var(--ink2); font-size: 13px; }
.legend i { display: inline-block; width: 22px; height: 12px; vertical-align: -1px; margin-right: 6px; }
.rowlabel { font-weight: 600; margin: 14px 0 6px; }
.panels { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
@media (max-width: 820px) { .panels { grid-template-columns: 1fr; } }
.panel { min-width: 0; position: relative; }
.panel svg { display: block; width: 100%; height: auto; }
.ptitle { font-size: 12px; color: var(--ink2); margin-bottom: 2px; }
svg text { font-family: var(--sans); font-size: 10.5px; fill: var(--muted); }
svg text.lab { fill: var(--ink2); font-size: 11px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
th, td { padding: 4px 10px; text-align: right; white-space: nowrap; }
th { color: var(--muted); font-weight: 500; border-bottom: 1px solid var(--axis); }
td:first-child, th:first-child, td.l, th.l { text-align: left; }
tr.sep td { border-top: 1px solid var(--grid); }
td.best { font-weight: 700; color: var(--ink); }
td.na { color: var(--muted); }
.gain { color: var(--good); }
#tip { position: fixed; pointer-events: none; background: var(--ink); color: var(--surface); padding: 6px 9px; border-radius: 6px;
  font-size: 12px; line-height: 1.35; z-index: 20; max-width: 240px; font-variant-numeric: tabular-nums; }
ul { margin: 6px 0 0; padding-left: 18px; color: var(--ink2); }
.note { font-size: 13px; }
</style>

<div class="wrap">
  <header>
    <h1>100M sweep: final loss over learning rate and beta / momentum</h1>
    <p class="sub" id="stamp"></p>
  </header>

  <div class="controls">
    <div class="ctl"><span>Optimizer</span><div class="seg" id="c-opt"></div></div>
    <div class="ctl"><span>Loss from</span><div class="seg" id="c-kind"></div></div>
    <div class="ctl"><span>Y range</span><div class="seg" id="c-zoom"></div></div>
  </div>

  <section class="card">
    <h2>Best setting per batch size and budget</h2>
    <p id="best-note"></p>
    <div class="scroll" style="margin-top:10px"><table id="best"></table></div>
  </section>

  <section class="card">
    <h2>Final loss against learning rate, one line per beta / momentum</h2>
    <p>Rows are token budgets, columns are batch sizes. With Y range "near the best" each panel has its own y axis (its best loss + 0.07), so read the tick labels before comparing batch sizes; "everything" uses one y axis per row. Hover a point for its value.</p>
    <div class="legend" id="legend"></div>
    <div id="grid"></div>
  </section>

  <section class="card">
    <h2>What a wrong beta / momentum costs</h2>
    <p>For each value, the loss at its best learning rate minus the loss of the best value in the same batch size and budget. Zero is the best value. The x axis is spaced by the averaging horizon 1 / (1 &minus; beta) in steps.</p>
    <div class="legend" id="legend2"></div>
    <div class="panels" id="cost" style="margin-top:8px"></div>
  </section>

  <section class="card">
    <h2>All numbers</h2>
    <p>Bold is the lowest loss in its batch size and budget. A dot is a setting that was not run; "run" is still training.</p>
    <div class="scroll" style="margin-top:10px"><table id="all"></table></div>
  </section>

  <section class="card note">
    <h2>What is and is not in here</h2>
    <ul id="notes"></ul>
  </section>
</div>
<div id="tip" hidden></div>

<script>
const DATA = __DATA__;
const OPTS = [["adam", "AdamW"], ["muon", "Muon"], ["muown_classical", "Muown"]];
const KINDS = [["train", "in-training validation"], ["fixed", "fixed validation set"]];
const ZOOMS = [["near", "near the best"], ["all", "everything"]];
const BUDGETS = ["1", "2", "4"], TOKENS = {"1": "2.0B", "2": "4.0B", "4": "8.1B"};
const HPS = ["0.9", "0.95", "0.975", "0.8"];            // fixed colour slot per value, never re-assigned
const COLOR = {"0.9": "var(--s1)", "0.95": "var(--s2)", "0.975": "var(--s3)", "0.8": "var(--s4)"};
const SHAPE = {"0.9": "circle", "0.95": "square", "0.975": "tri", "0.8": "diamond"};
const UNTIED = "0.9/0.95";
const NEAR = 0.07;                                       // "near the best": best loss of the row + this
const state = {opt: "muon", kind: "train", zoom: "near"};
try { Object.assign(state, JSON.parse(localStorage.getItem("sweep100m") || "{}")); } catch (e) {}

const $ = id => document.getElementById(id);
const fmtLr = v => { const e = Math.floor(Math.log10(v) + 1e-9), m = v / 10 ** e; return (Math.abs(m - Math.round(m)) < 1e-6 ? Math.round(m) : m.toFixed(1)) + "e" + e; };
const hpName = o => o === "adam" ? "beta1 = beta2" : "momentum";
const sel = () => DATA.rows.filter(r => r.opt === state.opt && r.kind === state.kind);
const uniq = a => [...new Set(a)];
const el = (tag, attrs, text) => { const n = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (text != null) n.textContent = text; return n; };

function segs(id, items, key) {
  const box = $(id); box.innerHTML = "";
  for (const [v, label] of items) {
    const b = document.createElement("button"); b.type = "button"; b.textContent = label; b.id = id + "-" + v;
    b.setAttribute("aria-pressed", String(state[key] === v));
    b.onclick = () => { state[key] = v; try { localStorage.setItem("sweep100m", JSON.stringify(state)); } catch (e) {} render(); };
    box.appendChild(b);
  }
}

function mark(shape, x, y, fill, r) {
  const a = {fill, stroke: "var(--surface)", "stroke-width": 2};
  if (shape === "square") return el("rect", {x: x - r, y: y - r, width: 2 * r, height: 2 * r, ...a});
  if (shape === "tri") return el("path", {d: `M${x},${y - r - 1}L${x + r + 1},${y + r}L${x - r - 1},${y + r}Z`, ...a});
  if (shape === "diamond") return el("path", {d: `M${x},${y - r - 1.5}L${x + r + 1.5},${y}L${x},${y + r + 1.5}L${x - r - 1.5},${y}Z`, ...a});
  return el("circle", {cx: x, cy: y, r, ...a});
}
function swatch(shape, fill, hollow) {
  const s = el("svg", {viewBox: "0 0 22 12", width: 22, height: 12, style: "vertical-align:-1px;margin-right:6px"});
  if (!hollow) s.appendChild(el("line", {x1: 0, x2: 22, y1: 6, y2: 6, stroke: fill, "stroke-width": 2}));
  const m = mark(shape, 11, 6, hollow ? "var(--surface)" : fill, 3.5);
  if (hollow) { m.setAttribute("stroke", fill); m.setAttribute("stroke-width", 1.6); }
  s.appendChild(m); return s;
}
function tipOn(node, html) {
  node.addEventListener("mouseenter", e => { const t = $("tip"); t.innerHTML = html; t.hidden = false; move(e); });
  node.addEventListener("mousemove", move);
  node.addEventListener("mouseleave", () => { $("tip").hidden = true; });
  function move(e) { const t = $("tip"); const w = t.offsetWidth, h = t.offsetHeight;
    t.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, e.clientX + 14)) + "px";
    t.style.top = Math.max(8, e.clientY - h - 12) + "px"; }
}
function ticks(lo, hi, n) {
  const raw = (hi - lo) / n, p = 10 ** Math.floor(Math.log10(raw)), f = raw / p;
  const step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p, out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(v);
  return [out, step];
}
function spread(labels, min, max, gap) {     // push direct labels apart vertically
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < gap) labels[i].y = labels[i - 1].y + gap;
  const over = labels.length ? labels[labels.length - 1].y - max : 0;
  if (over > 0) for (const l of labels) l.y -= over;
  for (let i = labels.length - 2; i >= 0; i--) if (labels[i + 1].y - labels[i].y < gap) labels[i].y = labels[i + 1].y - gap;
  for (const l of labels) l.y = Math.max(min, l.y);
}

function renderGrid(rows) {
  const lrs = uniq(rows.map(r => r.lr)).sort((a, b) => a - b), batches = uniq(rows.map(r => r.bs)).sort((a, b) => a - b);
  const hps = HPS.filter(h => rows.some(r => r.hp === h));
  const lg = $("legend"); lg.innerHTML = "";
  for (const h of hps) { const s = document.createElement("span"); s.appendChild(swatch(SHAPE[h], COLOR[h])); s.append(hpName(state.opt) + " " + h + (h === "0.95" ? " (phase 1)" : "")); lg.appendChild(s); }
  if (rows.some(r => r.hp === UNTIED)) { const s = document.createElement("span"); s.appendChild(swatch("diamond", "var(--ink2)", true)); s.append("beta1 0.9, beta2 0.95 (untied reference)"); lg.appendChild(s); }
  const grid = $("grid"); grid.innerHTML = "";
  const W = 360, H = 232, m = {l: 44, r: 46, t: 8, b: 30};
  const x = v => m.l + (lrs.length < 2 ? .5 : (Math.log(v) - Math.log(lrs[0])) / (Math.log(lrs[lrs.length - 1]) - Math.log(lrs[0]))) * (W - m.l - m.r);
  for (const bud of BUDGETS) {
    const rb = rows.filter(r => r.budget === bud); if (!rb.length) continue;
    // "near the best": each panel shows its own best loss + NEAR (the level differs a lot between batch
    // sizes, a shared range would push whole panels off the top); "everything": one range for the row.
    const range = rs => { const lo = Math.min(...rs.map(r => r.loss)), hiAll = Math.max(...rs.map(r => r.loss));
      const hi = state.zoom === "near" ? Math.min(hiAll, lo + NEAR) : hiAll, pad = (hi - lo) * .06 || .01; return [lo - pad, hi + pad]; };
    const cut = state.zoom === "near" ? batches.reduce((n, bs) => { const rs = rb.filter(r => r.bs === bs); return n + (rs.length ? rs.filter(r => r.loss > range(rs)[1]).length : 0); }, 0) : 0;
    const head = document.createElement("div"); head.className = "rowlabel";
    head.textContent = bud + "x budget (" + TOKENS[bud] + " tokens)" + (cut ? "  ·  " + cut + " point" + (cut > 1 ? "s" : "") + " above the range, switch Y range to see them" : "");
    grid.appendChild(head);
    const row = document.createElement("div"); row.className = "panels"; grid.appendChild(row);
    for (const bs of batches) {
      const rp = rb.filter(r => r.bs === bs), best = rp.length ? Math.min(...rp.map(r => r.loss)) : null;
      const [y0, y1] = range(state.zoom === "near" && rp.length ? rp : rb), y = v => m.t + (y1 - v) / (y1 - y0) * (H - m.t - m.b);
      const box = document.createElement("div"); box.className = "panel";
      const t = document.createElement("div"); t.className = "ptitle"; t.textContent = "batch " + bs; box.appendChild(t);
      const id = "clip-" + bud + "-" + bs;
      const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `final loss against learning rate, batch ${bs}, ${bud}x budget`});
      const defs = el("defs"), cp = el("clipPath", {id}); cp.appendChild(el("rect", {x: m.l - 6, y: m.t - 6, width: W - m.l - m.r + 12, height: H - m.t - m.b + 6})); defs.appendChild(cp); svg.appendChild(defs);
      const [tk, step] = ticks(y0, y1, 4), dec = step < .01 ? 3 : 2;
      for (const v of tk) { svg.appendChild(el("line", {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--grid)", "stroke-width": 1})); svg.appendChild(el("text", {x: m.l - 6, y: y(v) + 3.5, "text-anchor": "end"}, v.toFixed(dec))); }
      svg.appendChild(el("line", {x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1}));
      for (const v of lrs) svg.appendChild(el("text", {x: x(v), y: H - m.b + 14, "text-anchor": "middle"}, fmtLr(v)));
      svg.appendChild(el("text", {x: (m.l + W - m.r) / 2, y: H - 3, "text-anchor": "middle"}, "learning rate"));
      const g = el("g", {"clip-path": `url(#${id})`}); svg.appendChild(g);
      const labels = [];
      for (const h of hps) {
        const pts = rp.filter(r => r.hp === h).sort((a, b) => a.lr - b.lr); if (!pts.length) continue;
        if (pts.length > 1) g.appendChild(el("path", {d: pts.map((p, i) => (i ? "L" : "M") + x(p.lr) + "," + y(p.loss)).join(""), fill: "none", stroke: COLOR[h], "stroke-width": 2, "stroke-linejoin": "round"}));
        for (const p of pts) g.appendChild(mark(SHAPE[h], x(p.lr), y(p.loss), COLOR[h], 3.6));
        const last = pts[pts.length - 1]; if (last.loss <= y1) labels.push({y: y(last.loss), x: x(last.lr), text: h});
        for (const p of pts) { if (p.loss > y1) continue; const hit = el("circle", {cx: x(p.lr), cy: y(p.loss), r: 11, fill: "transparent"});
          tipOn(hit, `<b>${p.loss.toFixed(4)}</b><br>${hpName(state.opt)} ${h}, lr ${fmtLr(p.lr)}<br>batch ${bs}, ${bud}x<br>${(p.loss - best) < 5e-5 ? "best in this panel" : "+" + (p.loss - best).toFixed(4) + " over the panel's best"}`); svg.appendChild(hit); }
      }
      for (const p of rp.filter(r => r.hp === UNTIED)) { if (p.loss > y1) continue;
        const d = mark("diamond", x(p.lr), y(p.loss), "var(--surface)", 3.4); d.setAttribute("stroke", "var(--ink2)"); d.setAttribute("stroke-width", 1.6); g.appendChild(d);
        const hit = el("circle", {cx: x(p.lr), cy: y(p.loss), r: 9, fill: "transparent"});
        tipOn(hit, `<b>${p.loss.toFixed(4)}</b><br>beta1 0.9, beta2 0.95, lr ${fmtLr(p.lr)}<br>batch ${bs}, ${bud}x<br>${p.loss <= best ? "below every tied setting here" : "+" + (p.loss - best).toFixed(4) + " over the best tied setting"}`); svg.appendChild(hit); }
      spread(labels, m.t + 6, H - m.b - 2, 11);
      for (const l of labels) svg.appendChild(el("text", {x: Math.min(l.x + 9, W - m.r + 6), y: l.y + 3.5, class: "lab"}, l.text));
      if (!rp.length) svg.appendChild(el("text", {x: (m.l + W - m.r) / 2, y: H / 2, "text-anchor": "middle"}, "no results yet"));
      box.appendChild(svg); row.appendChild(box);
    }
  }
}

function bestAtHp(rows, bud, bs, h) { const v = rows.filter(r => r.budget === bud && r.bs === bs && r.hp === h); return v.length ? v.reduce((a, b) => a.loss <= b.loss ? a : b) : null; }

function renderCost(rows) {
  const batches = uniq(rows.map(r => r.bs)).sort((a, b) => a - b), hps = HPS.filter(h => rows.some(r => r.hp === h)).sort((a, b) => a - b);
  const shapes = ["circle", "square", "tri"], lg = $("legend2"); lg.innerHTML = "";
  batches.forEach((b, i) => { const s = document.createElement("span"); s.appendChild(swatch(shapes[i % 3], "var(--ink2)")); s.append("batch " + b); lg.appendChild(s); });
  const box = $("cost"); box.innerHTML = "";
  const W = 360, H = 220, m = {l: 44, r: 40, t: 8, b: 30}, hor = h => 1 / (1 - h);
  if (hps.length < 2) { box.textContent = "Needs at least two beta / momentum values."; return; }
  const x = h => m.l + (Math.log(hor(h)) - Math.log(hor(hps[0]))) / (Math.log(hor(hps[hps.length - 1])) - Math.log(hor(hps[0]))) * (W - m.l - m.r);
  const series = {}; let top = 0;
  for (const bud of BUDGETS) for (const bs of batches) {
    const pts = hps.map(h => ({h, b: bestAtHp(rows, bud, bs, h)})).filter(p => p.b); if (pts.length < 2) continue;
    const lo = Math.min(...pts.map(p => p.b.loss)); series[bud + "|" + bs] = pts.map(p => ({h: p.h, gap: p.b.loss - lo, lr: p.b.lr, loss: p.b.loss}));
    top = Math.max(top, ...series[bud + "|" + bs].map(p => p.gap));
  }
  const cap = state.zoom === "near" ? Math.min(top, NEAR) : top, y1 = (cap || .01) * 1.08, y = v => m.t + (y1 - v) / y1 * (H - m.t - m.b);
  for (const bud of BUDGETS) {
    const p = document.createElement("div"); p.className = "panel";
    const t = document.createElement("div"); t.className = "ptitle"; t.textContent = bud + "x budget"; p.appendChild(t);
    const id = "cclip-" + bud, svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `loss gap to the best beta or momentum, ${bud}x budget`});
    const defs = el("defs"), cp = el("clipPath", {id}); cp.appendChild(el("rect", {x: m.l - 6, y: m.t - 6, width: W - m.l - m.r + 12, height: H - m.t - m.b + 12})); defs.appendChild(cp); svg.appendChild(defs);
    const [tk, step] = ticks(0, y1, 4), dec = step < .01 ? 3 : 2;
    for (const v of tk) { svg.appendChild(el("line", {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: v === 0 ? "var(--axis)" : "var(--grid)", "stroke-width": 1})); svg.appendChild(el("text", {x: m.l - 6, y: y(v) + 3.5, "text-anchor": "end"}, "+" + v.toFixed(dec))); }
    for (const h of hps) svg.appendChild(el("text", {x: x(h), y: H - m.b + 14, "text-anchor": "middle"}, h));
    svg.appendChild(el("text", {x: (m.l + W - m.r) / 2, y: H - 3, "text-anchor": "middle"}, hpName(state.opt)));
    const g = el("g", {"clip-path": `url(#${id})`}); svg.appendChild(g); const labels = [];
    batches.forEach((bs, i) => { const s = series[bud + "|" + bs]; if (!s) return;
      g.appendChild(el("path", {d: s.map((q, j) => (j ? "L" : "M") + x(q.h) + "," + y(q.gap)).join(""), fill: "none", stroke: "var(--ink2)", "stroke-width": 2, "stroke-linejoin": "round"}));
      for (const q of s) { g.appendChild(mark(shapes[i % 3], x(q.h), y(q.gap), "var(--ink2)", 3.6)); if (q.gap > y1) continue;
        const hit = el("circle", {cx: x(q.h), cy: y(q.gap), r: 11, fill: "transparent"});
        tipOn(hit, `<b>${q.gap < 5e-5 ? "best value" : "+" + q.gap.toFixed(4)}</b><br>${hpName(state.opt)} ${q.h} at lr ${fmtLr(q.lr)}<br>batch ${bs}, ${bud}x: loss ${q.loss.toFixed(4)}`); svg.appendChild(hit); }
      const last = s[s.length - 1]; if (last.gap <= y1) labels.push({y: y(last.gap), x: x(last.h), text: String(bs)}); });
    spread(labels, m.t + 6, H - m.b - 2, 11);
    for (const l of labels) svg.appendChild(el("text", {x: l.x + 9, y: l.y + 3.5, class: "lab"}, l.text));
    if (!labels.length && !batches.some(bs => series[bud + "|" + bs])) svg.appendChild(el("text", {x: (m.l + W - m.r) / 2, y: H / 2, "text-anchor": "middle"}, "no results yet"));
    p.appendChild(svg); box.appendChild(p);
  }
}

function renderBest(rows) {
  const batches = uniq(rows.map(r => r.bs)).sort((a, b) => a - b), tied = rows.filter(r => r.hp !== UNTIED), t = $("best");
  let h = `<tr><th class="l">Budget</th><th>Batch</th><th>Best loss</th><th>lr</th><th>${hpName(state.opt)}</th><th>Best at 0.95</th><th>Gain over 0.95</th><th>Settings run</th></tr>`;
  for (const bud of BUDGETS) batches.forEach((bs, i) => {
    const c = tied.filter(r => r.budget === bud && r.bs === bs); if (!c.length) return;
    const b = c.reduce((a, q) => a.loss <= q.loss ? a : q), ref = bestAtHp(tied, bud, bs, "0.95"), gain = ref ? ref.loss - b.loss : null;
    h += `<tr class="${i ? "" : "sep"}"><td>${i ? "" : bud + "x"}</td><td>${bs}</td><td class="best">${b.loss.toFixed(4)}</td><td>${fmtLr(b.lr)}</td><td>${b.hp}</td><td>${ref ? ref.loss.toFixed(4) : "·"}</td><td class="${gain > 5e-5 ? "gain" : ""}">${gain == null ? "·" : gain < 5e-5 ? "0.95 is best" : "−" + gain.toFixed(4)}</td><td>${c.length}</td></tr>`;
  });
  t.innerHTML = h;
  $("best-note").textContent = "Lowest final loss among the tied settings that have finished, and how much it gains over the best learning rate at 0.95. One seed per setting: a gain of a few thousandths is within what a rerun could change.";
}

function renderAll(rows) {
  const lrs = uniq(rows.map(r => r.lr)).sort((a, b) => a - b), batches = uniq(rows.map(r => r.bs)).sort((a, b) => a - b);
  const hps = uniq(rows.map(r => r.hp)).sort((a, b) => (a === UNTIED) - (b === UNTIED) || parseFloat(a) - parseFloat(b));
  const run = new Set(DATA.unfinished.filter(u => u.opt === state.opt).map(u => [u.budget, u.bs, u.hp, u.lr].join("|")));
  let h = `<tr><th class="l">Budget</th><th>Batch</th><th class="l">${hpName(state.opt)}</th>${lrs.map(v => `<th>${fmtLr(v)}</th>`).join("")}</tr>`;
  for (const bud of BUDGETS) for (const bs of batches) {
    const c = rows.filter(r => r.budget === bud && r.bs === bs); if (!c.length) continue;
    const best = Math.min(...c.filter(r => r.hp !== UNTIED).map(r => r.loss));
    hps.forEach((hp, i) => { if (!c.some(r => r.hp === hp)) return;
      h += `<tr class="${i ? "" : "sep"}"><td>${i ? "" : bud + "x"}</td><td>${i ? "" : bs}</td><td class="l">${hp === UNTIED ? "0.9 / 0.95" : hp}</td>` + lrs.map(v => {
        const r = c.find(q => q.hp === hp && q.lr === v);
        if (r) return `<td class="${r.loss === best ? "best" : ""}">${r.loss.toFixed(4)}</td>`;
        return `<td class="na">${state.kind === "train" && run.has([bud, bs, hp, v].join("|")) ? "run" : "·"}</td>`; }).join("") + "</tr>"; });
  }
  $("all").innerHTML = h;
}

function renderNotes(rows) {
  const n = $("notes"), items = [], add = s => items.push(s), o = OPTS.find(x => x[0] === state.opt)[1];
  add(`${rows.length} finished results for ${o} from ${state.kind === "train" ? "the last validation of each cooldown (2,560 sequences; runs with the same batch size and budget are scored on the same sequences)" : "the scoring jobs on the fixed validation set (25,600 sequences, the same for every checkpoint)"}.`);
  if (state.kind === "fixed" && !rows.some(r => r.hp === "0.95")) add("The 0.95 column (phase 1) has no fixed-set scores yet: the rescoring jobs in ../rescore have not run. Use the in-training validation to compare with 0.95.");
  const un = DATA.unfinished.filter(u => u.opt === state.opt), ex = DATA.excluded.filter(u => u.opt === state.opt);
  if (un.length) add(`${un.length} cooldown${un.length > 1 ? "s" : ""} still running or without a final loss: ` + un.map(u => `${u.budget}x batch ${u.bs} lr ${fmtLr(u.lr)} ${u.hp}`).join("; ") + ".");
  if (ex.length) add(`${ex.length} cooldown${ex.length > 1 ? "s" : ""} left out because they did not branch from their checkpoint step and have no valid rerun: ` + ex.map(u => `${u.budget}x batch ${u.bs} lr ${fmtLr(u.lr)} ${u.hp} (${u.why})`).join("; ") + ".");
  add("A cooldown is used only if its log shows that it loaded exactly the step it asked for; a rerun replaces the first attempt.");
  add("Every setting is one run with one seed. Validation loss sits about 0.03 below training loss in all runs; that offset is the same everywhere and cancels in comparisons.");
  n.innerHTML = items.map(s => `<li>${s}</li>`).join("");
}

function render() {
  segs("c-opt", OPTS.filter(o => DATA.rows.some(r => r.opt === o[0])), "opt"); segs("c-kind", KINDS, "kind"); segs("c-zoom", ZOOMS, "zoom");
  const rows = sel();
  $("stamp").textContent = `100M-active hybrid MoE on JUPITER, WSD ladders at 1x / 2x / 4x budget. Read from the Slurm logs on ${DATA.generated}.`;
  renderBest(rows); renderGrid(rows); renderCost(rows.filter(r => r.hp !== UNTIED)); renderAll(rows); renderNotes(rows);
}
render();
</script>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", required=True, help="directory with the Slurm .out files")
    p.add_argument("--out", required=True, help="HTML file to write")
    a = p.parse_args()
    data = collect(a.logs)
    open(a.out, "w").write(PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    n = lambda k: sum(r["kind"] == k for r in data["rows"])
    print(f"wrote {a.out}: {n('train')} in-training results, {n('fixed')} fixed-set scores, "
          f"{len(data['unfinished'])} unfinished, {len(data['excluded'])} excluded")


if __name__ == "__main__":
    main()
