"""Build a static HTML dashboard for the MoE post-hoc sweep.

Scans results/moe_analysis/bsz*/nexp_*/lr*/120BT/ and emits:
  results/moe_analysis/dashboard/{index.html, style.css, app.js, manifest.json}

Pure static — open index.html directly or serve via any static file server.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT_DEFAULT = Path("results/moe_analysis")
LAYER_RE = re.compile(r"_layer_(\d+)\.")


def _layers_in(dir_path: Path, pattern: str) -> list[int]:
    if not dir_path.exists():
        return []
    layers = []
    for p in dir_path.glob(pattern):
        m = LAYER_RE.search(p.name)
        if m:
            layers.append(int(m.group(1)))
    return sorted(set(layers))


def _scan_bucket(bucket: Path) -> dict:
    coact_layers = _layers_in(bucket, "coactivation_layer_*.png")
    act_layers = _layers_in(bucket / "activation_norms_per_layer", "activation_norms_layer_*.png")
    gif_layers = _layers_in(bucket / "expert_routing_per_layer", "expert_routing_layer_*.gif")

    has = lambda rel: (bucket / rel).exists()
    return {
        "coactivation_layers": coact_layers,
        "activation_layers": act_layers,
        "routing_layers": gif_layers,
        "has_activation_norms_agg": has("expert_activation_norms.png"),
        "has_activation_max_over_median": has("expert_activation_max_over_median.png"),
        "has_routing_gif_agg": has("expert_routing.gif"),
        "has_saturation_plot": has("router_saturation_vs_final.png"),
        "has_saturation_json": has("saturation.json"),
        "has_coactivation_json": has("coactivation.json"),
        "has_token_counts_json": has("token_counts.json"),
        "has_activation_norms_json": has("expert_activation_norms.json"),
    }


def build_manifest(root: Path) -> dict:
    buckets: dict[str, dict[str, dict[str, dict]]] = {}
    bsz_set, nexp_set, lr_set = set(), set(), set()
    for bsz_dir in sorted(root.glob("bsz*")):
        if not bsz_dir.is_dir() or bsz_dir.name == "dashboard":
            continue
        bsz = bsz_dir.name.replace("bsz", "")
        for nexp_dir in sorted(bsz_dir.glob("nexp_*")):
            nexp = nexp_dir.name.replace("nexp_", "")
            for lr_dir in sorted(nexp_dir.glob("lr*")):
                lr = lr_dir.name.replace("lr", "")
                bucket = lr_dir / "120BT"
                if not bucket.is_dir():
                    continue
                bsz_set.add(bsz)
                nexp_set.add(nexp)
                lr_set.add(lr)
                buckets.setdefault(bsz, {}).setdefault(nexp, {})[lr] = _scan_bucket(bucket)

    def _sort_num(xs):
        return sorted(xs, key=lambda x: float(x))

    return {
        "bsz_values": _sort_num(bsz_set),
        "nexp_values": _sort_num(nexp_set),
        "lr_values": _sort_num(lr_set),
        "buckets": buckets,
        "stage": "120BT",
        "bucket_prefix": "../",
        "gif_ext": "gif",
        "no_raw_json": False,
    }


INDEX_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>MoE Post-Hoc Dashboard (120BT)</title>
<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
<link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap\">
<link rel=\"stylesheet\" href=\"style.css\">
</head>
<body>
<header>
  <h1><span class=\"brand\">MoE Post-Hoc Sweep</span><span class=\"brand-sep\">·</span><span class=\"brand-stage\">120BT</span></h1>
  <div class=\"selectors\">
    <label>Batch Size <select id=\"sel-bsz\"></select></label>
    <label>Number of Experts <select id=\"sel-nexp\"></select></label>
    <label>Learning Rate <select id=\"sel-lr\"></select></label>
    <span id=\"bucket-status\" class=\"status\"></span>
    <button id=\"theme-toggle\" class=\"theme-toggle\" type=\"button\" aria-label=\"Toggle dark mode\" title=\"Toggle dark mode\">
      <span class=\"theme-icon theme-icon-light\">☀</span>
      <span class=\"theme-icon theme-icon-dark\">☾</span>
    </button>
  </div>
</header>

<main>
  <section id=\"sec-saturation\" class=\"panel\">
    <h2><span class=\"sec-icon\">◎</span>Router Saturation</h2>
    <div id=\"saturation-content\"></div>
  </section>

  <section id=\"sec-activation\" class=\"panel\">
    <h2><span class=\"sec-icon\">∿</span>Expert Activation Norms</h2>
    <div class=\"agg-row\" id=\"activation-agg\"></div>
    <div class=\"sub\">
      <h3>Per-layer</h3>
      <div class=\"layer-tabs\" id=\"activation-tabs\"></div>
      <div id=\"activation-detail\" class=\"detail\"></div>
      <details>
        <summary>Browse all layers (scroll)</summary>
        <div class=\"hscroll\" id=\"activation-scroll\"></div>
      </details>
    </div>
  </section>

  <section id=\"sec-coactivation\" class=\"panel\">
    <h2><span class=\"sec-icon\">▦</span>Coactivation Matrices</h2>
    <div class=\"sub\">
      <h3>Selected layer</h3>
      <div class=\"layer-tabs\" id=\"coact-tabs\"></div>
      <div id=\"coact-detail\" class=\"detail\"></div>
      <details>
        <summary>Browse all layers (scroll)</summary>
        <div class=\"hscroll\" id=\"coact-scroll\"></div>
      </details>
    </div>
  </section>

  <section id=\"sec-routing\" class=\"panel\">
    <h2><span class=\"sec-icon\">⥃</span>Expert Routing <span class=\"h2-sub\">(per-layer, frame-stepped)</span></h2>
    <p class=\"hint\">Compare up to 3 layers side-by-side. Use Play/Pause, ◀/▶, or the slider to step through training checkpoints.</p>
    <div class=\"gif-compare\" id=\"gif-compare\"></div>
    <details>
      <summary>Aggregate routing GIF (first/mid/last)</summary>
      <div id=\"gif-agg\"></div>
    </details>
  </section>

  <section id=\"sec-raw\" class=\"panel\">
    <h2><span class=\"sec-icon\">{}</span>Raw JSON</h2>
    <ul id=\"raw-links\"></ul>
  </section>

  <footer class=\"footer\">
    <span>MoE Post-Hoc Dashboard</span>
    <span class=\"footer-sep\">·</span>
    <span id=\"footer-stats\"></span>
  </footer>
</main>

<script src=\"app.js\"></script>
</body>
</html>
"""

STYLE_CSS = """:root {
  --fg: #0f172a;
  --fg-soft: #334155;
  --muted: #64748b;
  --bg: #f1f5f9;
  --bg-grad-1: #eef2ff;
  --bg-grad-2: #faf5ff;
  --panel: #ffffff;
  --panel-soft: #f8fafc;
  --img-bg: #ffffff;
  --border: #e2e8f0;
  --border-strong: #cbd5e1;
  --accent: #4f46e5;
  --accent-soft: #eef2ff;
  --accent-strong: #4338ca;
  --danger: #dc2626;
  --danger-soft: #fef2f2;
  --danger-border: #fecaca;
  --header-bg: rgba(255, 255, 255, .85);
  --shadow-sm: 0 1px 2px rgba(15, 23, 42, .04);
  --shadow-md: 0 4px 12px -2px rgba(15, 23, 42, .08), 0 2px 4px -2px rgba(15, 23, 42, .04);
  --radius: 10px;
  --radius-sm: 6px;
}
:root[data-theme="dark"] {
  --fg: #e2e8f0;
  --fg-soft: #cbd5e1;
  --muted: #94a3b8;
  --bg: #0b1020;
  --bg-grad-1: #1e1b4b;
  --bg-grad-2: #312e81;
  --panel: #111827;
  --panel-soft: #0f172a;
  --img-bg: #f8fafc;
  --border: #1f2937;
  --border-strong: #334155;
  --accent: #818cf8;
  --accent-soft: #1e1b4b;
  --accent-strong: #c7d2fe;
  --danger: #f87171;
  --danger-soft: #2a1414;
  --danger-border: #7f1d1d;
  --header-bg: rgba(17, 24, 39, .85);
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, .35);
  --shadow-md: 0 4px 14px -2px rgba(0, 0, 0, .55), 0 2px 4px -2px rgba(0, 0, 0, .4);
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  color: var(--fg);
  background:
    radial-gradient(1200px 600px at 0% -10%, var(--bg-grad-1) 0%, transparent 60%),
    radial-gradient(900px 500px at 100% -10%, var(--bg-grad-2) 0%, transparent 55%),
    var(--bg);
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

/* Header */
header {
  padding: 1.1rem 1.75rem;
  background: var(--header-bg);
  backdrop-filter: saturate(180%) blur(12px);
  -webkit-backdrop-filter: saturate(180%) blur(12px);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 10;
}
header h1 {
  margin: 0 0 .65rem;
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: -0.015em;
  display: flex;
  align-items: center;
  gap: .55rem;
}
header h1::before {
  content: '';
  width: 10px; height: 10px;
  border-radius: 999px;
  background: linear-gradient(135deg, var(--accent), #a855f7);
  box-shadow: 0 0 0 3px var(--accent-soft), 0 0 12px rgba(79, 70, 229, .35);
}
.brand {
  background: linear-gradient(120deg, var(--accent) 0%, #a855f7 50%, #ec4899 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  color: transparent;
}
.brand-sep { color: var(--border-strong); font-weight: 400; margin: 0 .15rem; }
.brand-stage {
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .12em;
  text-transform: uppercase;
  padding: .2rem .55rem;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent-strong);
  font-variant-numeric: tabular-nums;
}
.selectors {
  display: flex;
  gap: .85rem;
  align-items: center;
  flex-wrap: wrap;
}
.selectors label {
  font-size: .78rem;
  font-weight: 500;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .04em;
  display: inline-flex;
  align-items: center;
  gap: .4rem;
}
.selectors select {
  margin-left: 0;
  padding: .4rem 1.85rem .4rem .75rem;
  font-size: .85rem;
  font-weight: 500;
  font-family: inherit;
  color: var(--fg);
  background-color: var(--panel);
  background-image: linear-gradient(45deg, transparent 50%, var(--muted) 50%), linear-gradient(135deg, var(--muted) 50%, transparent 50%);
  background-position: calc(100% - 14px) 50%, calc(100% - 9px) 50%;
  background-size: 5px 5px;
  background-repeat: no-repeat;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow-sm);
  cursor: pointer;
  appearance: none;
  -webkit-appearance: none;
  transition: border-color .15s, box-shadow .15s, transform .1s;
}
.selectors select:hover { border-color: var(--accent); transform: translateY(-1px); }
.selectors select:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
.status {
  font-size: .78rem;
  color: var(--accent-strong);
  margin-left: auto;
  padding: .35rem .8rem;
  background: linear-gradient(135deg, var(--accent-soft), color-mix(in srgb, var(--accent-soft) 60%, transparent));
  border: 1px solid color-mix(in srgb, var(--accent) 18%, transparent);
  border-radius: 999px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  letter-spacing: .01em;
}

/* Main layout */
main {
  padding: 1.5rem 1.75rem 3rem;
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
  max-width: 1600px;
  margin: 0 auto;
}

/* Panel */
@keyframes panel-in {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.5rem 1.5rem;
  box-shadow: var(--shadow-sm);
  transition: box-shadow .25s, transform .25s, border-color .25s;
  animation: panel-in .35s ease both;
  position: relative;
}
.panel::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: var(--radius);
  padding: 1px;
  background: linear-gradient(135deg, transparent 40%, rgba(79, 70, 229, .25) 100%);
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  opacity: 0;
  transition: opacity .25s;
  pointer-events: none;
}
.panel:hover { box-shadow: var(--shadow-md); }
.panel:hover::before { opacity: 1; }
.panel h2 {
  margin: 0 0 1rem;
  font-size: 1rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  display: flex;
  align-items: center;
  gap: .55rem;
  padding-bottom: .75rem;
  border-bottom: 1px solid var(--border);
}
.sec-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  font-size: .85rem;
  font-weight: 600;
  border-radius: 7px;
  background: linear-gradient(135deg, var(--accent-soft), color-mix(in srgb, var(--accent-soft) 70%, transparent));
  color: var(--accent-strong);
  border: 1px solid color-mix(in srgb, var(--accent) 18%, transparent);
}
.h2-sub { font-weight: 400; color: var(--muted); font-size: .82rem; margin-left: .2rem; }
.panel h3 {
  margin: .75rem 0 .5rem;
  font-size: .8rem;
  font-weight: 500;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .05em;
}
.sub { margin-top: 1rem; }
.hint { font-size: .82rem; color: var(--muted); margin: 0 0 .65rem; }

/* Aggregate row */
.agg-row {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
}
.agg-row img {
  max-width: 100%;
  max-height: 320px;
  width: auto;
  height: auto;
  margin: 0 auto;
  object-fit: contain;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--img-bg);
  display: block;
}

/* Layer tabs (segmented control style) */
.layer-tabs {
  display: flex;
  gap: .25rem;
  flex-wrap: wrap;
  margin-bottom: .85rem;
  padding: .3rem;
  background: linear-gradient(180deg, var(--panel-soft), color-mix(in srgb, var(--panel-soft) 70%, var(--bg)));
  border: 1px solid var(--border);
  border-radius: 8px;
}
.layer-tabs button {
  font-family: inherit;
  font-size: .78rem;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
  padding: .3rem .65rem;
  border: 1px solid transparent;
  background: transparent;
  color: var(--fg-soft);
  cursor: pointer;
  border-radius: 4px;
  transition: background-color .12s, color .12s, border-color .12s;
}
.layer-tabs button:hover {
  background: var(--panel);
  color: var(--fg);
  border-color: var(--border-strong);
}
.layer-tabs button.active {
  background: linear-gradient(135deg, var(--accent), #a855f7);
  color: #fff;
  border-color: transparent;
  box-shadow: 0 2px 6px rgba(79, 70, 229, .35), inset 0 1px 0 rgba(255, 255, 255, .15);
}

/* Detail */
.detail {
  background:
    radial-gradient(600px 200px at 50% 0%, color-mix(in srgb, var(--accent-soft) 60%, transparent), transparent 70%),
    var(--panel-soft);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 1rem;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 240px;
}
.detail img {
  max-width: 100%;
  max-height: 420px;
  width: auto;
  height: auto;
  object-fit: contain;
  border-radius: 6px;
  background: var(--img-bg);
  box-shadow: 0 2px 10px rgba(15, 23, 42, .08);
  transition: transform .2s, box-shadow .2s;
}
.detail img:hover { transform: scale(1.01); box-shadow: 0 6px 24px rgba(15, 23, 42, .14); }
#saturation-content img {
  max-width: 100%;
  max-height: 380px;
  width: auto;
  height: auto;
  object-fit: contain;
  display: block;
  margin: 0 auto;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--img-bg);
}
#gif-agg img {
  max-width: 100%;
  max-height: 360px;
  width: auto;
  height: auto;
  object-fit: contain;
  display: block;
  margin: 0 auto;
}

/* Horizontal scroll strip */
.hscroll {
  display: flex;
  gap: .65rem;
  overflow-x: auto;
  padding: .75rem;
  background: var(--panel-soft);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  scroll-snap-type: x proximity;
  scrollbar-width: thin;
  scrollbar-color: var(--border-strong) transparent;
}
.hscroll::-webkit-scrollbar { height: 8px; }
.hscroll::-webkit-scrollbar-track { background: transparent; }
.hscroll::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }
.hscroll figure {
  flex: 0 0 auto;
  margin: 0;
  text-align: center;
  scroll-snap-align: start;
}
.hscroll img {
  height: 220px;
  width: auto;
  border: 1px solid var(--border);
  border-radius: 4px;
  display: block;
  background: var(--img-bg);
  transition: transform .12s, box-shadow .12s;
}
.hscroll img:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
.hscroll figcaption {
  font-size: .72rem;
  color: var(--muted);
  margin-top: .35rem;
  font-variant-numeric: tabular-nums;
  letter-spacing: .03em;
}

/* GIF compare */
.gif-compare {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: .85rem;
}
.gif-controls {
  display: flex;
  align-items: center;
  gap: .65rem;
  flex-wrap: wrap;
  padding: .55rem .75rem;
  background: linear-gradient(180deg, var(--panel-soft), color-mix(in srgb, var(--panel-soft) 70%, var(--bg)));
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  margin-bottom: .25rem;
}
.gif-controls .step-btn {
  font-family: inherit;
  font-size: .9rem;
  font-weight: 500;
  width: 32px;
  height: 32px;
  padding: 0;
  cursor: pointer;
  border: 1px solid var(--border-strong);
  background: var(--panel);
  color: var(--fg-soft);
  border-radius: var(--radius-sm);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: background-color .12s, border-color .12s, color .12s;
}
.gif-controls .step-btn:hover:not(:disabled) {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
.gif-controls .step-btn:disabled { opacity: .4; cursor: not-allowed; }
.gif-controls .play-btn { width: auto; padding: 0 .85rem; gap: .35rem; }
.gif-controls input[type=range] {
  flex: 1;
  min-width: 140px;
  accent-color: var(--accent);
}
.gif-controls .frame-count {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: .78rem;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  min-width: 70px;
  text-align: right;
}
.gif-controls .speed-sel {
  font-family: inherit;
  font-size: .78rem;
  padding: .25rem .4rem;
  border: 1px solid var(--border-strong);
  border-radius: 4px;
  background: var(--panel);
  color: var(--fg);
}
.sync-btn {
  font-family: inherit;
  font-size: .82rem;
  font-weight: 500;
  padding: .4rem .85rem;
  cursor: pointer;
  border: 1px solid var(--border-strong);
  background: var(--panel);
  color: var(--fg-soft);
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow-sm);
  transition: background-color .12s, border-color .12s, color .12s;
  display: inline-flex;
  align-items: center;
  gap: .35rem;
}
.sync-btn:hover {
  background: linear-gradient(135deg, var(--accent), #a855f7);
  color: #fff;
  border-color: transparent;
  box-shadow: 0 2px 8px rgba(79, 70, 229, .35);
}
@keyframes spin-once { from { transform: rotate(0); } to { transform: rotate(360deg); } }
.sync-btn:active span:first-child { display: inline-block; animation: spin-once .5s ease; }
.gif-slot {
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: .75rem;
  background: linear-gradient(180deg, var(--panel-soft), color-mix(in srgb, var(--panel-soft) 80%, var(--bg)));
  display: flex;
  flex-direction: column;
  gap: .6rem;
  transition: transform .2s, box-shadow .2s, border-color .2s;
}
.gif-slot:hover {
  transform: translateY(-2px);
  border-color: color-mix(in srgb, var(--accent) 35%, var(--border));
  box-shadow: var(--shadow-md);
}
.gif-slot select {
  width: 100%;
  font-family: inherit;
  font-size: .82rem;
  padding: .3rem .5rem;
  border: 1px solid var(--border-strong);
  border-radius: 4px;
  background: var(--panel);
  color: var(--fg);
  cursor: pointer;
}
.gif-slot select:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
.gif-slot img,
.gif-slot canvas {
  width: 100%;
  max-width: 100%;
  max-height: 320px;
  height: auto;
  margin: 0 auto;
  object-fit: contain;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--img-bg);
  display: block;
}
.gif-slot .slot-status {
  font-size: .72rem;
  color: var(--muted);
  text-align: center;
  font-variant-numeric: tabular-nums;
  min-height: 1em;
}

/* Details / summary */
details {
  margin-top: 1rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--panel-soft);
}
details[open] { background: var(--panel); }
details summary {
  cursor: pointer;
  font-size: .85rem;
  font-weight: 500;
  color: var(--accent-strong);
  padding: .65rem .85rem;
  user-select: none;
  list-style: none;
  display: flex;
  align-items: center;
  gap: .4rem;
}
details summary::-webkit-details-marker { display: none; }
details summary::before {
  content: '▸';
  font-size: .7rem;
  transition: transform .15s;
  color: var(--muted);
}
details[open] summary::before { transform: rotate(90deg); }
details > *:not(summary) { padding: 0 .85rem .85rem; }

/* Raw links */
#raw-links {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-wrap: wrap;
  gap: .5rem;
}
#raw-links li { margin: 0; }
#raw-links a {
  display: inline-block;
  padding: .35rem .75rem;
  font-size: .82rem;
  font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  color: var(--accent-strong);
  background: var(--accent-soft);
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  text-decoration: none;
  transition: background-color .12s, border-color .12s;
}
#raw-links a:hover { background: var(--panel); border-color: var(--accent); }

/* Missing state */
.missing {
  color: var(--danger);
  font-style: italic;
  font-size: .82rem;
  padding: .5rem .75rem;
  background: var(--danger-soft);
  border: 1px dashed var(--danger-border);
  border-radius: var(--radius-sm);
  display: inline-block;
}

/* Theme toggle */
.theme-toggle {
  margin-left: .25rem;
  width: 34px;
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  cursor: pointer;
  background: var(--panel);
  color: var(--fg-soft);
  border: 1px solid var(--border-strong);
  border-radius: 999px;
  box-shadow: var(--shadow-sm);
  transition: background-color .15s, color .15s, border-color .15s, transform .15s;
  position: relative;
  overflow: hidden;
}
.theme-toggle:hover {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
.theme-toggle:active { transform: scale(0.94); }
.theme-icon { line-height: 1; transition: opacity .2s, transform .25s; position: absolute; }
:root[data-theme="dark"] .theme-icon-light { opacity: 0; transform: rotate(-90deg) scale(.6); }
:root[data-theme="dark"] .theme-icon-dark { opacity: 1; transform: rotate(0) scale(1); }
:root:not([data-theme="dark"]) .theme-icon-light { opacity: 1; transform: rotate(0) scale(1); }
:root:not([data-theme="dark"]) .theme-icon-dark { opacity: 0; transform: rotate(90deg) scale(.6); }

/* Footer */
.footer {
  margin-top: .5rem;
  padding: 1rem 1.5rem;
  font-size: .78rem;
  color: var(--muted);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: .65rem;
  font-variant-numeric: tabular-nums;
}
.footer-sep { color: var(--border-strong); }
#footer-stats { font-family: 'JetBrains Mono', ui-monospace, monospace; }

@media (max-width: 1000px) {
  .agg-row { grid-template-columns: 1fr; }
  .gif-compare { grid-template-columns: 1fr; }
}
@media (max-width: 600px) {
  header, main { padding-left: 1rem; padding-right: 1rem; }
  .status { margin-left: 0; }
}
"""

APP_JS = r"""(async function() {
  // Theme handling: persisted in localStorage, defaults to system preference.
  const THEME_KEY = 'moe-dashboard-theme';
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const initialTheme = localStorage.getItem(THEME_KEY) || (prefersDark ? 'dark' : 'light');
  document.documentElement.dataset.theme = initialTheme;
  const themeBtn = document.getElementById('theme-toggle');
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      localStorage.setItem(THEME_KEY, next);
    });
  }

  const M = await fetch('manifest.json').then(r => r.json());
  const STAGE = M.stage;

  const $ = id => document.getElementById(id);
  const selBsz = $('sel-bsz'), selNexp = $('sel-nexp'), selLr = $('sel-lr');

  const BUCKET_PREFIX = M.bucket_prefix !== undefined ? M.bucket_prefix : '../';
  const GIF_EXT = M.gif_ext || 'gif';
  const HIDE_RAW_JSON = !!M.no_raw_json;

  function bucketPath(bsz, nexp, lr) {
    return `${BUCKET_PREFIX}bsz${bsz}/nexp_${nexp}/lr${lr}/${STAGE}`;
  }

  function fillSelect(sel, values, prev) {
    sel.innerHTML = '';
    for (const v of values) {
      const opt = document.createElement('option');
      opt.value = v; opt.textContent = v;
      sel.appendChild(opt);
    }
    if (prev && values.includes(prev)) sel.value = prev;
  }

  function currentBucket() {
    const bsz = selBsz.value, nexp = selNexp.value, lr = selLr.value;
    const b = ((M.buckets[bsz] || {})[nexp] || {})[lr];
    return b ? { bsz, nexp, lr, info: b, path: bucketPath(bsz, nexp, lr) } : null;
  }

  function refreshNexp() {
    const bsz = selBsz.value;
    const nexps = Object.keys(M.buckets[bsz] || {}).sort((a,b) => +a - +b);
    fillSelect(selNexp, nexps, selNexp.value);
    refreshLr();
  }
  function refreshLr() {
    const bsz = selBsz.value, nexp = selNexp.value;
    const lrs = Object.keys((M.buckets[bsz] || {})[nexp] || {}).sort((a,b) => +a - +b);
    fillSelect(selLr, lrs, selLr.value);
    render();
  }

  function buildLayerTabs(container, layers, onSelect, initial) {
    container.innerHTML = '';
    layers.forEach((L, idx) => {
      const btn = document.createElement('button');
      btn.textContent = `L${L}`;
      btn.dataset.layer = L;
      btn.addEventListener('click', () => {
        container.querySelectorAll('button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        onSelect(L);
      });
      if ((initial !== undefined && L === initial) || (initial === undefined && idx === 0)) {
        btn.classList.add('active');
      }
      container.appendChild(btn);
    });
  }

  function renderSaturation(b) {
    const el = $('saturation-content');
    el.innerHTML = '';
    if (b.info.has_saturation_plot) {
      const img = document.createElement('img');
      img.src = `${b.path}/router_saturation_vs_final.png`;
      img.alt = 'router_saturation_vs_final';
      img.style.maxWidth = '100%';
      el.appendChild(img);
    } else {
      el.innerHTML = '<div class="missing">No saturation plot.</div>';
    }
  }

  function renderActivation(b) {
    const agg = $('activation-agg');
    agg.innerHTML = '';
    if (b.info.has_activation_norms_agg) {
      const img = document.createElement('img');
      img.src = `${b.path}/expert_activation_norms.png`;
      img.alt = 'expert_activation_norms';
      agg.appendChild(img);
    }
    if (b.info.has_activation_max_over_median) {
      const img = document.createElement('img');
      img.src = `${b.path}/expert_activation_max_over_median.png`;
      img.alt = 'expert_activation_max_over_median';
      agg.appendChild(img);
    }
    if (!agg.children.length) agg.innerHTML = '<div class="missing">No aggregate plots.</div>';

    const layers = b.info.activation_layers;
    const tabs = $('activation-tabs');
    const detail = $('activation-detail');
    const scroll = $('activation-scroll');
    scroll.innerHTML = '';
    detail.innerHTML = '';

    if (!layers.length) {
      tabs.innerHTML = '<span class="missing">No per-layer data.</span>';
      return;
    }
    const showLayer = (L) => {
      detail.innerHTML = '';
      const img = document.createElement('img');
      img.src = `${b.path}/activation_norms_per_layer/activation_norms_layer_${L}.png`;
      img.alt = `activation_norms_layer_${L}`;
      detail.appendChild(img);
    };
    buildLayerTabs(tabs, layers, showLayer);
    showLayer(layers[0]);
    layers.forEach(L => {
      const fig = document.createElement('figure');
      const img = document.createElement('img');
      img.src = `${b.path}/activation_norms_per_layer/activation_norms_layer_${L}.png`;
      img.loading = 'lazy';
      const cap = document.createElement('figcaption'); cap.textContent = `L${L}`;
      fig.appendChild(img); fig.appendChild(cap);
      scroll.appendChild(fig);
    });
  }

  function renderCoactivation(b) {
    const layers = b.info.coactivation_layers;
    const tabs = $('coact-tabs');
    const detail = $('coact-detail');
    const scroll = $('coact-scroll');
    scroll.innerHTML = '';
    detail.innerHTML = '';
    if (!layers.length) {
      tabs.innerHTML = '<span class="missing">No coactivation plots.</span>';
      return;
    }
    const showLayer = (L) => {
      detail.innerHTML = '';
      const img = document.createElement('img');
      img.src = `${b.path}/coactivation_layer_${L}.png`;
      img.alt = `coactivation_layer_${L}`;
      detail.appendChild(img);
    };
    buildLayerTabs(tabs, layers, showLayer);
    showLayer(layers[0]);
    layers.forEach(L => {
      const fig = document.createElement('figure');
      const img = document.createElement('img');
      img.src = `${b.path}/coactivation_layer_${L}.png`;
      img.loading = 'lazy';
      const cap = document.createElement('figcaption'); cap.textContent = `L${L}`;
      fig.appendChild(img); fig.appendChild(cap);
      scroll.appendChild(fig);
    });
  }

  // Decoded-frame cache: url -> Promise<{frames: ImageBitmap[], w, h}>.
  const _frameCache = new Map();

  async function loadFrames(url) {
    if (_frameCache.has(url)) return _frameCache.get(url);
    const p = (async () => {
      if (typeof ImageDecoder === 'undefined') {
        throw new Error('ImageDecoder not supported in this browser');
      }
      const buf = await fetch(url).then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.arrayBuffer();
      });
      const type = url.endsWith('.webp') ? 'image/webp' : 'image/gif';
      const decoder = new ImageDecoder({ data: buf, type });
      await decoder.tracks.ready;
      const track = decoder.tracks.selectedTrack;
      const targetCount = Number.isFinite(track.frameCount) && track.frameCount > 0 ? track.frameCount : 4096;
      const frames = [];
      let w = 0, h = 0;
      for (let i = 0; i < targetCount; i++) {
        try {
          const result = await decoder.decode({ frameIndex: i });
          if (!w) { w = result.image.displayWidth; h = result.image.displayHeight; }
          const bmp = await createImageBitmap(result.image);
          result.image.close();
          frames.push(bmp);
        } catch (e) {
          break;
        }
      }
      decoder.close();
      if (!frames.length) throw new Error('No frames decoded');
      return { frames, w, h };
    })();
    _frameCache.set(url, p);
    p.catch(() => _frameCache.delete(url));
    return p;
  }

  // Master timeline shared across the 3 routing slots.
  const _master = {
    slots: [],          // [{canvas, ctx, status, frames, w, h, layer, url}]
    frame: 0,
    playing: true,
    fps: 8,
    raf: null,
    lastTickAt: 0,
    slider: null,
    counter: null,
    playBtn: null,
  };

  function maxFrames() {
    return _master.slots.reduce((m, s) => Math.max(m, s.frames ? s.frames.length : 0), 0);
  }

  function drawSlot(s) {
    if (!s.frames || !s.frames.length) return;
    const idx = Math.min(_master.frame, s.frames.length - 1);
    const cv = s.canvas, ctx = s.ctx;
    if (cv.width !== s.w || cv.height !== s.h) { cv.width = s.w; cv.height = s.h; }
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(s.frames[idx], 0, 0);
  }

  function renderAllSlots() {
    _master.slots.forEach(drawSlot);
    const total = maxFrames();
    if (_master.slider) _master.slider.max = Math.max(0, total - 1);
    if (_master.slider) _master.slider.value = _master.frame;
    if (_master.counter) _master.counter.textContent = total ? `${_master.frame + 1} / ${total}` : '— / —';
  }

  function tick(now) {
    if (!_master.playing) { _master.raf = null; return; }
    if (!_master.lastTickAt) _master.lastTickAt = now;
    const dt = now - _master.lastTickAt;
    const interval = 1000 / Math.max(1, _master.fps);
    if (dt >= interval) {
      _master.lastTickAt = now;
      const total = maxFrames();
      if (total > 0) {
        _master.frame = (_master.frame + 1) % total;
        renderAllSlots();
      }
    }
    _master.raf = requestAnimationFrame(tick);
  }

  function startPlayback() {
    if (_master.playing && _master.raf) return;
    _master.playing = true;
    _master.lastTickAt = 0;
    if (_master.playBtn) _master.playBtn.textContent = '❚❚ Pause';
    if (_master.raf) cancelAnimationFrame(_master.raf);
    _master.raf = requestAnimationFrame(tick);
  }
  function pausePlayback() {
    _master.playing = false;
    if (_master.playBtn) _master.playBtn.textContent = '▶ Play';
    if (_master.raf) { cancelAnimationFrame(_master.raf); _master.raf = null; }
  }

  async function loadSlot(s) {
    s.frames = null;
    s.status.textContent = 'loading…';
    drawSlot(s);
    try {
      const data = await loadFrames(s.url());
      s.frames = data.frames;
      s.w = data.w;
      s.h = data.h;
      s.status.textContent = `${data.frames.length} frames`;
      renderAllSlots();
    } catch (e) {
      s.status.textContent = `error: ${e.message}`;
      console.error('loadSlot', e);
    }
  }

  function renderRouting(b) {
    const layers = b.info.routing_layers;
    const compare = $('gif-compare');
    compare.innerHTML = '';
    pausePlayback();
    _master.slots = [];
    _master.frame = 0;

    if (!layers.length) {
      compare.innerHTML = '<div class="missing">No per-layer routing animations.</div>';
    } else {
      // Master controls bar (spans full row above the 3 slots).
      const controls = document.createElement('div');
      controls.className = 'gif-controls';
      controls.style.gridColumn = '1 / -1';

      const playBtn = document.createElement('button');
      playBtn.className = 'step-btn play-btn';
      playBtn.textContent = '❚❚ Pause';
      playBtn.title = 'Play / pause';
      playBtn.addEventListener('click', () => { _master.playing ? pausePlayback() : startPlayback(); });

      const prevBtn = document.createElement('button');
      prevBtn.className = 'step-btn';
      prevBtn.textContent = '◀';
      prevBtn.title = 'Step back';
      prevBtn.addEventListener('click', () => {
        pausePlayback();
        const total = maxFrames();
        if (total) { _master.frame = (_master.frame - 1 + total) % total; renderAllSlots(); }
      });

      const nextBtn = document.createElement('button');
      nextBtn.className = 'step-btn';
      nextBtn.textContent = '▶';
      nextBtn.title = 'Step forward';
      nextBtn.addEventListener('click', () => {
        pausePlayback();
        const total = maxFrames();
        if (total) { _master.frame = (_master.frame + 1) % total; renderAllSlots(); }
      });

      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = 0; slider.max = 0; slider.value = 0;
      slider.title = 'Scrub frame';
      slider.addEventListener('input', () => {
        pausePlayback();
        _master.frame = +slider.value;
        renderAllSlots();
      });

      const counter = document.createElement('span');
      counter.className = 'frame-count';
      counter.textContent = '— / —';

      const speedSel = document.createElement('select');
      speedSel.className = 'speed-sel';
      speedSel.title = 'Playback speed (fps)';
      [4, 8, 12, 20, 30].forEach(f => {
        const o = document.createElement('option');
        o.value = f; o.textContent = `${f} fps`;
        if (f === _master.fps) o.selected = true;
        speedSel.appendChild(o);
      });
      speedSel.addEventListener('change', () => { _master.fps = +speedSel.value; });

      controls.appendChild(playBtn);
      controls.appendChild(prevBtn);
      controls.appendChild(nextBtn);
      controls.appendChild(slider);
      controls.appendChild(counter);
      controls.appendChild(speedSel);
      compare.appendChild(controls);

      _master.slider = slider;
      _master.counter = counter;
      _master.playBtn = playBtn;

      const defaults = [
        layers[0],
        layers[Math.floor(layers.length / 2)],
        layers[layers.length - 1],
      ];
      defaults.forEach((defL) => {
        const slot = document.createElement('div'); slot.className = 'gif-slot';
        const sel = document.createElement('select');
        layers.forEach(L => {
          const o = document.createElement('option');
          o.value = L; o.textContent = `Layer ${L}`;
          if (L === defL) o.selected = true;
          sel.appendChild(o);
        });
        const canvas = document.createElement('canvas');
        const status = document.createElement('div');
        status.className = 'slot-status';
        const slotState = {
          canvas,
          ctx: canvas.getContext('2d'),
          status,
          frames: null,
          w: 0, h: 0,
          layer: defL,
          url: () => `${b.path}/expert_routing_per_layer/expert_routing_layer_${slotState.layer}.${GIF_EXT}`,
        };
        sel.addEventListener('change', () => {
          slotState.layer = sel.value;
          loadSlot(slotState);
        });
        slot.appendChild(sel);
        slot.appendChild(canvas);
        slot.appendChild(status);
        compare.appendChild(slot);
        _master.slots.push(slotState);
      });

      // Kick off all three loads in parallel; start playback once any is ready.
      Promise.all(_master.slots.map(loadSlot)).then(() => {
        if (maxFrames() > 0) startPlayback();
      });
    }

    const agg = $('gif-agg'); agg.innerHTML = '';
    if (b.info.has_routing_gif_agg) {
      const img = document.createElement('img');
      img.src = `${b.path}/expert_routing.${GIF_EXT}`;
      img.alt = 'expert_routing';
      img.style.maxWidth = '100%';
      agg.appendChild(img);
    } else {
      agg.innerHTML = '<div class="missing">No aggregate animation.</div>';
    }
  }

  function renderRaw(b) {
    const ul = $('raw-links');
    ul.innerHTML = '';
    if (HIDE_RAW_JSON) {
      const sec = document.getElementById('sec-raw');
      if (sec) sec.style.display = 'none';
      return;
    }
    const files = [
      ['saturation.json', b.info.has_saturation_json],
      ['coactivation.json', b.info.has_coactivation_json],
      ['token_counts.json', b.info.has_token_counts_json],
      ['expert_activation_norms.json', b.info.has_activation_norms_json],
    ];
    for (const [f, ok] of files) {
      if (!ok) continue;
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = `${b.path}/${f}`; a.textContent = f; a.target = '_blank';
      li.appendChild(a);
      ul.appendChild(li);
    }
  }

  function render() {
    const b = currentBucket();
    const status = $('bucket-status');
    if (!b) { status.textContent = 'No bucket'; return; }
    status.textContent = `bsz${b.bsz} / nexp_${b.nexp} / lr${b.lr} / ${STAGE}`;
    renderSaturation(b);
    renderActivation(b);
    renderCoactivation(b);
    renderRouting(b);
    renderRaw(b);
  }

  // Footer stats
  const totalBuckets = Object.values(M.buckets).reduce(
    (n, ne) => n + Object.values(ne).reduce((m, lr) => m + Object.keys(lr).length, 0), 0);
  const fs = document.getElementById('footer-stats');
  if (fs) fs.textContent = `${totalBuckets} buckets · ${M.bsz_values.length} bsz × ${M.nexp_values.length} nexp × ${M.lr_values.length} lr`;

  fillSelect(selBsz, M.bsz_values);
  refreshNexp();
  selBsz.addEventListener('change', refreshNexp);
  selNexp.addEventListener('change', refreshLr);
  selLr.addEventListener('change', render);
})();
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT,
                        help="results/moe_analysis directory (scanned for buckets)")
    parser.add_argument("--out", type=Path, default=None,
                        help="dashboard output dir (default: <root>/dashboard)")
    parser.add_argument("--publish-dir", type=Path, default=None,
                        help="When set, write dashboard files at the ROOT of this dir (paths become bszX/... not ../bszX/...). Overrides --out.")
    parser.add_argument("--gif-ext", choices=["gif", "webp"], default="gif",
                        help="Extension to use for routing animations (publish flow uses 'webp').")
    parser.add_argument("--no-raw-json", action="store_true",
                        help="Hide the Raw JSON panel (use when JSONs are not staged).")
    args = parser.parse_args()

    root = args.root.resolve()
    if args.publish_dir is not None:
        out = args.publish_dir.resolve()
        bucket_prefix = ""
    else:
        out = (args.out or (root / "dashboard")).resolve()
        bucket_prefix = "../"
    out.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(root)
    manifest["bucket_prefix"] = bucket_prefix
    manifest["gif_ext"] = args.gif_ext
    manifest["no_raw_json"] = bool(args.no_raw_json)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "index.html").write_text(INDEX_HTML)
    (out / "style.css").write_text(STYLE_CSS)
    (out / "app.js").write_text(APP_JS)

    n_buckets = sum(len(lr) for nexp in manifest["buckets"].values() for lr in nexp.values())
    print(f"Wrote dashboard to {out}")
    print(f"  bsz: {manifest['bsz_values']}")
    print(f"  nexp: {manifest['nexp_values']}")
    print(f"  lr: {manifest['lr_values']}")
    print(f"  buckets: {n_buckets}")
    print(f"  bucket_prefix: '{bucket_prefix}'  gif_ext: {args.gif_ext}  no_raw_json: {args.no_raw_json}")


if __name__ == "__main__":
    main()
